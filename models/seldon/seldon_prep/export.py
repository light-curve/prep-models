"""Export SELDON encoder to ONNX.

Strategy: the model uses a Python for-loop over observations, calling
a torchode adaptive ODE solver at each step.  torchode is not ONNX-
exportable, so we split into three ONNX files and a thin glue loop:

  seldon_embed.onnx    – per-observation embedding (flux + band → hidden)
  seldon_ode_step.onnx – single ODE step h(t0) → h(t1), fixed RK4
  seldon_gru_cell.onnx – GRU cell update: (x_embedded, h) → h_new
  seldon_deepset.onnx  – DeepSet aggregation over ODE-decoded states

The glue loop (Python / C++ / any language):

    h = zeros(B, hidden_dim)
    for t in range(T-1, 0, -1):
        t0 = scaled_time[:, t]           # (B,)
        t1 = scaled_time[:, t-1]         # (B,)
        same_time = (t1 == t0)
        t1_adj = t1 - DT * same_time     # avoid t0==t1 singularity
        h = ode_step(h, t0, t1_adj)      # seldon_ode_step.onnx
        h = gru_cell(x_emb[:, t-1], h,   # seldon_gru_cell.onnx
                     mask[:, t-1])

    # aggregation phase
    zs = ode_step_grid(h, t_grid)        # seldon_ode_step.onnx over grid
    zs_t = concat(zs, t_grid)            # append scaled time
    z = deepset(zs_t)                    # seldon_deepset.onnx

Note: RK4 replaces the trained adaptive Tsit5 solver.  For well-resolved
light curves the difference is negligible; the ODE is autonomous so one
RK4 step per observation interval is usually sufficient.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Make the upstream TAE package importable
_CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


# ---------------------------------------------------------------------------
# Wrapper modules (ONNX-exportable)
# ---------------------------------------------------------------------------


class EmbedStep(nn.Module):
    """Embed a single time step: flux + band_idx → hidden vector."""

    def __init__(self, encoder):
        super().__init__()
        self.band_emb = encoder.band_emb
        self.flux_embedding = encoder.flux_embedding
        self.linear_layer = encoder.linear_layer
        self.time_scaling = encoder.time_scaling
        self.softplus = encoder.softplus

    def forward(
        self,
        flux: torch.Tensor,  # (B, 1)  raw flux value
        band_idx: torch.Tensor,  # (B,)  integer band index
        time: torch.Tensor,  # (B,)  raw time value
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_embedded, scaled_time)  both (B, hidden_dim) / (B,)."""
        t_scale = self.softplus(self.time_scaling)
        scaled_t = time * t_scale

        f = flux.unsqueeze(-1) if flux.dim() == 1 else flux  # (B, 1)
        emb = self.flux_embedding(f)  # (B, hidden_dim)
        e_b = self.band_emb(band_idx)  # (B, BAND_EMB_DIM)
        x = torch.cat([emb, e_b], dim=-1)  # (B, input_dim + BAND_EMB_DIM)
        x = self.linear_layer(x)  # (B, hidden_dim)
        return x, scaled_t


class RK4ODEStep(nn.Module):
    """Single 4th-order Runge-Kutta step for the autonomous ODE dy/dt = f(y).

    Replaces torchode's adaptive Tsit5 solver with a fixed, ONNX-traceable
    integration step.  The ODE function (a ResNet) is time-independent, so
    passing t=None to it is fine.
    """

    def __init__(self, ode_func: nn.Module):
        super().__init__()
        self.ode_func = ode_func  # ODEFunc wrapping a ResNet

    def forward(
        self,
        h: torch.Tensor,  # (B, hidden_dim)
        t0: torch.Tensor,  # (B,) start times
        t1: torch.Tensor,  # (B,) end times
    ) -> torch.Tensor:  # (B, hidden_dim)
        dt = (t1 - t0).unsqueeze(-1)  # (B, 1)
        k1 = self.ode_func(None, h)
        k2 = self.ode_func(None, h + 0.5 * dt * k1)
        k3 = self.ode_func(None, h + 0.5 * dt * k2)
        k4 = self.ode_func(None, h + dt * k3)
        return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class GRUCellStep(nn.Module):
    """GRU cell update with observation mask.

    Applies the GRU cell when the observation is valid (mask=True),
    otherwise passes h through unchanged.
    """

    def __init__(self, gru_cell: nn.GRUCell):
        super().__init__()
        self.gru_cell = gru_cell

    def forward(
        self,
        x: torch.Tensor,  # (B, hidden_dim) embedded observation
        h: torch.Tensor,  # (B, hidden_dim) current hidden state
        mask: torch.Tensor,  # (B,) bool, True = valid observation
    ) -> torch.Tensor:
        h_new = self.gru_cell(x, h)
        return torch.where(mask.unsqueeze(-1), h_new, h)


class DeepSetAggregator(nn.Module):
    """Wraps SimpleDeepSet for the aggregation phase."""

    def __init__(self, deepset: nn.Module):
        super().__init__()
        self.deepset = deepset

    def forward(self, zs_t: torch.Tensor) -> torch.Tensor:
        """zs_t: (B, N_points, hidden_dim + 1)  → z: (B, hidden_dim)"""
        return self.deepset(zs_t)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_encoder(checkpoint_dir: Path, device: str = "cpu"):
    """Load the SELDON encoder from a Lightning checkpoint directory."""
    from omegaconf import OmegaConf
    from TAE.datasets.dataset_factory import get_module as get_dataset
    from TAE.experiments.factory import get_module as get_experiment
    from TAE.models.factory import get_module
    from TAE.util import fill_config

    config_file = checkpoint_dir / "hparams.yaml"
    weights_file = next(checkpoint_dir.glob("checkpoints/epoch*"))

    cfg = OmegaConf.load(config_file)
    cfg["dataset"]["config"]["num_workers"] = 0
    cfg["dataset"]["config"]["batch_size"] = 1
    cfg = fill_config(cfg)
    config = OmegaConf.to_container(cfg)

    model = get_module(cfg["model_params"]["name"], config["model_params"]["config"])
    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])
    experiment = get_experiment(
        cfg["exp_params"]["name"],
        {"model": model, "data": data, **config["exp_params"]["config"]},
    )

    experiment.load_from_checkpoint(
        checkpoint_path=weights_file,
        model=model,
        data=data,
        params=cfg["exp_params"]["config"]["params"],
    )
    experiment.eval()
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def run_export(out_dir: Path, checkpoint_dir: Path, device: str = "cpu") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_dir} ...")
    model = load_encoder(checkpoint_dir, device=device)
    encoder = model.encoder  # RecurrentODEEncoder

    hidden_dim = encoder.hidden_dim
    B = 2  # example batch size for tracing

    # -- 1. Embed step --
    embed_module = EmbedStep(encoder).eval()
    dummy_flux = torch.zeros(B, 1, device=device)
    dummy_band = torch.zeros(B, dtype=torch.long, device=device)
    dummy_time = torch.zeros(B, device=device)

    embed_path = out_dir / "seldon_embed.onnx"
    torch.onnx.export(
        embed_module,
        (dummy_flux, dummy_band, dummy_time),
        str(embed_path),
        input_names=["flux", "band_idx", "time"],
        output_names=["x_embedded", "scaled_time"],
        dynamic_axes={
            "flux": {0: "batch"},
            "band_idx": {0: "batch"},
            "time": {0: "batch"},
            "x_embedded": {0: "batch"},
            "scaled_time": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported {embed_path}")

    # -- 2. ODE step --
    ode_func = encoder.rnn.ode_net  # ODEFunc (ResNet)
    ode_step_module = RK4ODEStep(ode_func).eval()
    dummy_h = torch.zeros(B, hidden_dim, device=device)
    dummy_t0 = torch.zeros(B, device=device)
    dummy_t1 = torch.ones(B, device=device) * 0.01

    ode_path = out_dir / "seldon_ode_step.onnx"
    torch.onnx.export(
        ode_step_module,
        (dummy_h, dummy_t0, dummy_t1),
        str(ode_path),
        input_names=["h", "t0", "t1"],
        output_names=["h_next"],
        dynamic_axes={
            "h": {0: "batch"},
            "t0": {0: "batch"},
            "t1": {0: "batch"},
            "h_next": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported {ode_path}")

    # -- 3. GRU cell --
    gru_cell_module = GRUCellStep(encoder.rnn.rnn_cell).eval()
    dummy_x_emb = torch.zeros(B, hidden_dim, device=device)
    dummy_mask = torch.ones(B, dtype=torch.bool, device=device)

    gru_path = out_dir / "seldon_gru_cell.onnx"
    torch.onnx.export(
        gru_cell_module,
        (dummy_x_emb, dummy_h, dummy_mask),
        str(gru_path),
        input_names=["x_embedded", "h", "mask"],
        output_names=["h_next"],
        dynamic_axes={
            "x_embedded": {0: "batch"},
            "h": {0: "batch"},
            "mask": {0: "batch"},
            "h_next": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported {gru_path}")

    # -- 4. DeepSet aggregator --
    N_points = encoder.N_points
    deepset_module = DeepSetAggregator(encoder.deepset).eval()
    # Input: (B, N_points, hidden_dim + 1) — hidden states concat with scaled time
    dummy_zs_t = torch.zeros(B, N_points, hidden_dim + 1, device=device)

    deepset_path = out_dir / "seldon_deepset.onnx"
    torch.onnx.export(
        deepset_module,
        (dummy_zs_t,),
        str(deepset_path),
        input_names=["zs_t"],
        output_names=["z"],
        dynamic_axes={
            "zs_t": {0: "batch"},
            "z": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported {deepset_path}")

    print(f"\nAll ONNX files written to {out_dir}")
    print(f"  hidden_dim  = {hidden_dim}")
    print(f"  N_points    = {N_points}")
    print(f"  t_max       = {encoder.t_max}")
    print(f"  NUM_BANDS   = {encoder.NUM_BANDS}")
