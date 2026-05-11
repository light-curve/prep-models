"""Export AstroM3 photo encoder to ONNX."""

from __future__ import annotations

import types
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from astrom3_prep.config import ENC_IN, OUTPUT_PREFIX, SEQ_LEN
from astrom3_prep.runtime import add_code_to_path, load_model


def _patch_model_for_onnx(model: nn.Module) -> None:
    """Replace ProbAttention with SDPA for ONNX tracing compatibility.

    ProbAttention uses torch.randint (random key sampling) and advanced
    in-place scatter updates that the TorchScript-based ONNX tracer cannot
    handle.  We replace it with standard scaled dot-product attention, which
    is equivalent in expectation and fully ONNX-exportable.

    The upstream Informer uses mask_flag=False for the encoder, so the
    ProbAttention causal mask is not needed; we omit it in the replacement.
    """
    for layer in model.encoder.attn_layers:
        _patch_attn_layer(layer.attention)


def _patch_attn_layer(attn_layer: nn.Module) -> None:
    def _sdpa_forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # queries/keys/values arrive as (B, L, H, D) from AttentionLayer
        q = queries.permute(0, 2, 1, 3)  # (B, H, L, D)
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.permute(0, 2, 1, 3).contiguous(), None

    attn_layer.inner_attention.forward = types.MethodType(
        _sdpa_forward, attn_layer.inner_attention
    )


class _AstroM3Embedder(nn.Module):
    """ONNX wrapper exposing per-timestep Informer encoder features.

    Accesses enc_embedding, encoder, and dropout directly so we can return
    all three aggregations in a single forward pass without the flattening
    and classification head from the original Informer.forward().
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.enc_embedding = model.enc_embedding
        self.encoder = model.encoder
        self.dropout = model.dropout

    def forward(
        self,
        x_enc: torch.Tensor,  # (batch, seq_len, enc_in)
        mask: torch.Tensor,  # (batch, seq_len) float32 — 1=valid, 0=pad
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_out = self.enc_embedding(x_enc)  # (B, 200, 128)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # (B, 200, 128)
        enc_out = self.dropout(enc_out)

        mask_f = mask.unsqueeze(-1)  # (B, 200, 1)

        mean_out = (enc_out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)

        neg_inf = torch.full_like(enc_out, -1e9)
        max_out = torch.where(mask_f > 0.5, enc_out, neg_inf).max(1).values

        sequence_out = enc_out  # unmasked; user applies mask when needed

        return mean_out, max_out, sequence_out


def load_export_model() -> nn.Module:
    add_code_to_path()
    model = load_model()
    _patch_model_for_onnx(model)
    return model


def run_export(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading AstroM3 photo encoder ...")
    model = load_export_model()

    dummy = (
        torch.zeros(2, SEQ_LEN, ENC_IN, dtype=torch.float32),
        torch.ones(2, SEQ_LEN, dtype=torch.float32),
    )
    input_names = ["x_enc", "mask"]
    output_names = ["mean", "max", "sequence"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    wrapper = _AstroM3Embedder(model)
    wrapper.eval()

    out_path = output_dir / f"{OUTPUT_PREFIX}.onnx"
    print(f"Exporting {out_path.name} (mean, max, sequence) ...")

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        dynamo=False,
    )

    proto = onnx.load(str(out_path))
    for out in proto.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output '{out.name}': {dims}")

    print("Export complete.")
