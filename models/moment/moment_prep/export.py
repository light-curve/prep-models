"""Export a MOMENT-1 encoder to ONNX.

The ONNX model takes a batch of univariate time series (magnitudes, NaN for
padding/missing) over MOMENT's fixed 512-step context window and returns the
encoder patch embeddings.  MOMENT has no timestamp input, so only magnitude
values are passed (the model's own RevIN instance normalization handles
scaling), consistent with how this repo feeds the Chronos models.  See the
README for the time / irregular-sampling discussion.

Input
-----
context : float32 [batch, 512]
    Magnitude values.  NaN marks padded / missing observations.  The sequence
    length is fixed at 512 (MOMENT's native context); only ``batch`` is dynamic.

Outputs
-------
mean     : float32 [batch, d_model]
    Masked mean pool over valid context patches (MOMENT's "embedding" output).
sequence : float32 [batch, 64, d_model]
    Per-patch encoder hidden states (one row per patch of 8 observations).
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn

from momentfm import MOMENTPipeline

from moment_prep.config import PATCH_LEN, SEQ_LEN, hf_repo, hf_revision, output_prefix


def _load_model(size: str) -> MOMENTPipeline:
    # Pin the HF revision so the loaded weights are 100% reproducible.
    model = MOMENTPipeline.from_pretrained(
        hf_repo(size),
        revision=hf_revision(size),
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.eval()
    return model


class _MomentEmbedder(nn.Module):
    """ONNX-exportable MOMENT encoder wrapper.

    Reimplements ``MOMENT.embed`` (reduction="mean") with deterministic,
    ONNX-friendly ops: the masked RevIN statistics are computed with
    ``torch.where`` masking instead of ``nanmean``, and the ``unfold`` patch
    views are replaced by plain reshapes (valid because stride == patch_len).
    The patch-embedding and T5 encoder submodules are reused unchanged, so the
    output matches the upstream model to floating-point precision.

    The mask is derived from the input itself (NaN = padding/missing), giving a
    single-tensor interface consistent with the Chronos exports in this repo.
    """

    def __init__(self, model: MOMENTPipeline) -> None:
        super().__init__()
        self.patch_len: int = int(model.patch_len)
        self.d_model: int = int(model.config.d_model)
        self.norm_eps: float = float(model.normalizer.eps)

        # Reused submodules (weights unchanged).
        self.patch_embedding = model.patch_embedding
        self.encoder = model.encoder

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # context: (batch, seq), float32, NaN = padding / missing.
        batch, seq = context.shape
        P = self.patch_len
        n_patches = seq // P

        valid = (~torch.isnan(context)).float()  # (B, L)
        x = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 1. RevIN masked instance normalisation (over valid positions) ---
        count = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        mean = (x * valid).sum(dim=-1, keepdim=True) / count
        residual = (x - mean) * valid
        var = (residual * residual).sum(dim=-1, keepdim=True) / count
        std = var.sqrt() + self.norm_eps
        x = (x - mean) / std
        x = x * valid  # zero padded positions (matches upstream nan_to_num)

        # --- 2. Patch view (stride == patch_len, so unfold == reshape) ---
        # x_patched: (B, n_channels=1, n_patches, P)
        x_patched = x.view(batch, 1, n_patches, P)
        # A patch counts as observed only if all P positions are valid.
        patch_valid = (valid.view(batch, n_patches, P).sum(dim=-1) == P).float()

        # --- 3. Patch embedding (reimplemented around the reused weights) ---
        pe = self.patch_embedding
        value_embed = pe.value_embedding(x_patched)  # (B, 1, n_patches, d_model)
        pmask = patch_valid.view(batch, 1, n_patches, 1)  # (B, 1, n_patches, 1)
        enc_in = pmask * value_embed + (1.0 - pmask) * pe.mask_embedding
        if pe.add_positional_embedding:
            enc_in = enc_in + pe.position_embedding(enc_in)
        enc_in = enc_in.reshape(batch, n_patches, self.d_model)  # n_channels == 1

        # --- 4. T5 encoder ---
        enc_out = self.encoder(
            inputs_embeds=enc_in,
            attention_mask=patch_valid,  # (B, n_patches)
        ).last_hidden_state  # (B, n_patches, d_model)

        # --- 5. Pool over valid patches (channels collapsed; n_channels == 1) ---
        sequence = enc_out  # (B, n_patches, d_model)
        pv = patch_valid.unsqueeze(-1)  # (B, n_patches, 1)
        mean_emb = (sequence * pv).sum(dim=1) / pv.sum(dim=1).clamp(min=1.0)

        return mean_emb, sequence


def run_export(output_dir: Path, size: str) -> None:
    print(f"Loading {hf_repo(size)} ...")
    model = _load_model(size)

    embedder = _MomentEmbedder(model)
    embedder.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed seq length; left-pad first sample with NaN to exercise masking.
    dummy = torch.zeros(2, SEQ_LEN, dtype=torch.float32)
    dummy[0, : SEQ_LEN // 2] = float("nan")

    dynamic_axes = {
        "context": {0: "batch"},
        "mean": {0: "batch"},
        "sequence": {0: "batch"},
    }

    out_path = output_dir / f"{output_prefix(size)}.onnx"
    print(f"Exporting {out_path.name}  (fixed seq={SEQ_LEN}, patch={PATCH_LEN}) ...")
    with torch.no_grad():
        torch.onnx.export(
            embedder,
            (dummy,),
            str(out_path),
            input_names=["context"],
            output_names=["mean", "sequence"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            dynamo=False,
        )

    proto = onnx.load(str(out_path))
    for out in proto.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  output '{out.name}': {dims}")
    print(f"  d_model={embedder.d_model}")
    print("Export complete.")
