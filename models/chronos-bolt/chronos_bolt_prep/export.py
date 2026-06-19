"""Export a Chronos-Bolt encoder to ONNX.

The ONNX model takes a batch of univariate time series (magnitudes, NaN for
padding) and returns encoder patch embeddings.  The interface and preprocessing
are identical to the Chronos 2 export: following the StarEmbed benchmarking
approach, timestamps are not used — only magnitude values are passed, and the
model's own instance normalization handles scaling.

Input
-----
context : float32 [batch, seq]
    Magnitude values.  NaN marks left-padded / missing observations.
    Both axes are dynamic; ``seq`` must be a multiple of the patch size (16),
    up to the model's native context of 2048.

Outputs
-------
mean     : float32 [batch, d_model]
    Masked mean pool over valid context patches.
sequence : float32 [batch, num_patches, d_model]
    Per-patch encoder hidden states (context patches only, [REG] excluded).
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn

from chronos import ChronosBoltPipeline

from chronos_bolt_prep.config import hf_repo, hf_revision, output_prefix


def _load_pipeline(size: str) -> ChronosBoltPipeline:
    # Pin the HF revision so the loaded weights are 100% reproducible.
    return ChronosBoltPipeline.from_pretrained(
        hf_repo(size), revision=hf_revision(size), device_map="cpu"
    )


class _ChronosBoltEmbedder(nn.Module):
    """ONNX-exportable Chronos-Bolt encoder wrapper.

    Reimplements ``ChronosBoltModelForForecasting.encode`` with deterministic,
    ONNX-friendly ops (no nanmean / arcsinh / token-id gather), exposing the
    same ``context -> (mean, sequence)`` interface as the Chronos 2 export.
    """

    def __init__(self, pipeline: ChronosBoltPipeline) -> None:
        super().__init__()
        model = pipeline.model
        self.patch_size: int = model.chronos_config.input_patch_size
        self.use_arcsinh: bool = bool(model.instance_norm.use_arcsinh)
        self.norm_eps: float = float(model.instance_norm.eps)

        # Model submodules
        self.input_patch_embedding = model.input_patch_embedding
        self.encoder = model.encoder

        # Pre-computed [REG] token embedding; shape (1, 1, d_model).
        # Precomputing avoids the int64 token-id Gather that breaks ONNX export.
        reg_id = model.config.reg_token_id
        with torch.no_grad():
            reg_embed = model.shared(torch.tensor([[reg_id]], dtype=torch.long))
        self.register_buffer("reg_embed", reg_embed.detach())  # (1, 1, d_model)

    def _instance_norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked mean/std normalisation, ONNX-compatible (no nanmean)."""
        # x: (batch, seq), NaN = padding
        valid = ~torch.isnan(x)
        zeros = torch.zeros_like(x)
        x_safe = torch.where(valid, x, zeros)  # NaN -> 0 (NaN * 0.0 != 0 in IEEE 754)

        count = valid.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        loc = x_safe.sum(dim=-1, keepdim=True) / count

        residual = torch.where(valid, x - loc, zeros)
        var = (residual * residual).sum(dim=-1, keepdim=True) / count
        scale = var.sqrt()
        # Replace zero scale with eps
        scale = scale + (scale == 0).float() * self.norm_eps

        normed = torch.where(valid, (x_safe - loc) / scale, zeros)
        if self.use_arcsinh:
            # arcsinh(x) = log(x + sqrt(x^2 + 1)) — avoids aten::asinh which has
            # no TorchScript ONNX symbolic registered in any opset.
            normed = torch.log(normed + torch.sqrt(normed * normed + 1.0))

        return normed, valid

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # context: (batch, seq), float32, NaN = left-padded / missing.
        # seq must be a multiple of patch_size; it is a dynamic ONNX axis.
        batch = context.shape[0]
        P = self.patch_size

        # --- 1. Instance normalisation (masked positions become 0) ---
        normed, valid = self._instance_norm(context)  # (batch, seq)

        # --- 2. Patching: reshape to (batch, num_patches, patch_size) ---
        patches = normed.view(batch, -1, P)  # (B, K, P), K dynamic
        mask_patches = valid.float().view(batch, -1, P)

        # --- 3. Concatenate [values | mask] -> input to patch embedding ---
        patched_context = torch.cat([patches, mask_patches], dim=-1)  # (B, K, 2P)
        input_embeds = self.input_patch_embedding(patched_context)  # (B, K, d_model)

        # --- 4. Append [REG] token ---
        reg = self.reg_embed.expand(batch, -1, -1)  # (B, 1, d_model)
        input_embeds = torch.cat([input_embeds, reg], dim=1)  # (B, K+1, d_model)

        # --- 5. Attention mask: 1 for patches with any valid obs + 1 for [REG] ---
        patch_valid = mask_patches.sum(dim=-1) > 0  # (B, K)
        reg_ones = torch.ones(batch, 1, dtype=torch.float32, device=context.device)
        attention_mask = torch.cat([patch_valid.float(), reg_ones], dim=1)  # (B, K+1)

        # --- 6. Encoder ---
        encoder_out = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden = encoder_out[0]  # (B, K+1, d_model)

        # --- 7. Pool over context patches only (drop the [REG] token) ---
        sequence = hidden[:, :-1, :]  # (B, K, d_model)
        patch_valid_f = patch_valid.float().unsqueeze(-1)  # (B, K, 1)
        mean = (sequence * patch_valid_f).sum(dim=1) / patch_valid_f.sum(dim=1).clamp(
            min=1.0
        )

        return mean, sequence


def run_export(output_dir: Path, size: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {hf_repo(size)} ...")
    pipeline = _load_pipeline(size)
    pipeline.model.eval()

    embedder = _ChronosBoltEmbedder(pipeline)
    embedder.eval()

    d_model: int = pipeline.model.model_dim

    # The sequence axis is dynamic; this length is only the concrete example
    # used to trace the export (any multiple of patch_size 16 works).
    trace_len = 32 * embedder.patch_size  # 512
    dummy_context = torch.zeros(2, trace_len, dtype=torch.float32)
    # Left-pad first sample with NaN to exercise the masking path
    dummy_context[0, : trace_len // 2] = float("nan")

    dynamic_axes = {
        "context": {0: "batch", 1: "seq"},
        "mean": {0: "batch"},
        "sequence": {0: "batch", 1: "num_patches"},
    }

    out_path = output_dir / f"{output_prefix(size)}.onnx"
    print(f"Exporting {out_path.name}  (dynamic seq; trace length={trace_len}) ...")

    with torch.no_grad():
        torch.onnx.export(
            embedder,
            (dummy_context,),
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
    print(f"  d_model={d_model}, patch_size={embedder.patch_size}")
    print("Export complete.")
