"""Export ATAT LC encoder to a single ONNX file with three named outputs.

Input convention (ONNX):
    data  [batch, seq_len, 6]  – per-band flux/magnitude
    time  [batch, seq_len, 6]  – per-band observation times
    mask  [batch, seq_len, 6]  – 1 = valid observation, 0 = padding

The mask convention matches the ATAT internal convention (1=valid), so no
inversion is needed here.

Outputs:
    token    [batch, 192]          – CLS token
    mean     [batch, 192]          – masked mean pool
    sequence [batch, seq_len*6, 192] – per-observation features (excl. CLS)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from atat_prep.config import (
    CODE_DIR,
    DATASET_CHANNEL,
    SEQ_LEN,
    T_MAX,
    WEIGHTS_DIR,
)


def _add_code_to_path() -> None:
    code = str(CODE_DIR)
    if code not in sys.path:
        sys.path.insert(0, code)
    # Prevent Python from writing .pyc files into the read-only submodule tree.
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def load_encoder() -> nn.Module:
    """Load just the Encoder from the checkpoint state dict.

    Avoids importing main_model / datasets / utils from the upstream training
    stack (which drags in ray, h5py, pandas, etc.).  Instead we load the
    checkpoint with torch.load, extract hyper_parameters, instantiate Encoder
    directly, and copy only the encoder weights.
    """
    _add_code_to_path()
    from layers.encoder_ellastic import Encoder  # noqa: PLC0415

    ckpt = WEIGHTS_DIR / "checkpoint.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt}. Run 'prep-models atat download' first."
        )
    print(f"Loading checkpoint from {ckpt} ...")
    raw = torch.load(str(ckpt), map_location="cpu")

    # PL stores all constructor kwargs under 'hyper_parameters'.
    # T_max and dataset_channel are computed at training time by utils.update_config
    # from the dataset registry (datasets.py), so they are NOT saved in the checkpoint.
    # We inject them from our known ELASTICC constants.
    hp = dict(raw["hyper_parameters"])
    hp.setdefault("T_max", T_MAX)
    hp.setdefault("dataset_channel", DATASET_CHANNEL)
    hp.setdefault("n_classes", 20)  # ELASTICC has 20 transient classes

    encoder = Encoder(**hp)
    encoder.eval()

    # The PL state dict prefixes encoder weights with "E."
    enc_state = {
        k[len("E.") :]: v for k, v in raw["state_dict"].items() if k.startswith("E.")
    }
    encoder.load_state_dict(enc_state)
    return encoder


class _ATATEmbedder(nn.Module):
    """Wraps the ATAT Encoder and returns all three aggregation variants in one pass."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        data: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # data/time/mask: [batch, seq_len, dataset_channel]
        # mask: 1=valid, 0=padding (same as ATAT internal convention)
        emb_x, sorted_mask = self.encoder.time_modulator(data, time, mask)
        # emb_x:       [batch, seq_len*dataset_channel, embed_dim]
        # sorted_mask: [batch, seq_len*dataset_channel, 1]

        has_token = self.encoder.emb_to_classifier == "token"
        if has_token:
            bs = emb_x.shape[0]
            token_rep = self.encoder.token.expand(bs, 1, -1)
            mask_token = torch.ones(bs, 1, 1, device=emb_x.device)
            full_mask = torch.cat([mask_token, sorted_mask], dim=1)
            emb_x = torch.cat([token_rep, emb_x], dim=1)
        else:
            full_mask = sorted_mask

        emb_x = self.encoder.transformer(emb_x, full_mask)

        embedding_token = emb_x[:, 0, :]
        seq_emb = emb_x[:, 1:, :] if has_token else emb_x
        denom = sorted_mask.sum(dim=1).clamp(min=1.0)
        embedding_mean = (seq_emb * sorted_mask).sum(dim=1) / denom
        # seq_emb excludes the CLS token, so "sequence" and "token" are non-overlapping
        embedding_sequence = seq_emb

        return embedding_token, embedding_mean, embedding_sequence


def run_export(output_dir: Path) -> None:
    """Export all aggregation variants to a single ONNX file ``atat.onnx``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = load_encoder()
    wrapper = _ATATEmbedder(encoder)
    wrapper.eval()

    batch = 2
    dummy = (
        torch.zeros(batch, SEQ_LEN, DATASET_CHANNEL),
        torch.zeros(batch, SEQ_LEN, DATASET_CHANNEL),
        torch.ones(batch, SEQ_LEN, DATASET_CHANNEL),
    )

    input_names = ["data", "time", "mask"]
    output_names = ["token", "mean", "sequence"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    out_path = output_dir / "atat.onnx"
    print("Exporting atat.onnx ...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        dynamo=False,
    )
    print(f"  Saved: {out_path}")

    proto = onnx.load(str(out_path))
    for out in proto.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output '{out.name}': {dims}")
    print("Export complete.")
