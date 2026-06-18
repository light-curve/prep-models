"""Export Chronos 2 encoder to ONNX.

The ONNX model takes a batch of univariate time series (magnitudes, NaN for
padding) and returns encoder patch embeddings.

Following the StarEmbed benchmarking approach, timestamps are not used: the
model treats observations as sequentially ordered, so the preprocessing step
is just magnitude normalization (handled internally by Chronos 2's instance
normalization).

Input
-----
context : float32 [batch, CONTEXT_LENGTH]
    Magnitude values.  NaN marks left-padded / missing observations.

Outputs
-------
mean     : float32 [batch, d_model]
    Masked mean pool over valid context patches.
sequence : float32 [batch, num_patches, d_model]
    Per-patch encoder hidden states (context patches only, [REG] and
    output-patch tokens excluded).
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn

from chronos import Chronos2Pipeline

from chronos2_prep.config import CONTEXT_LENGTH, HF_REPO, OUTPUT_PREFIX


def _load_pipeline() -> Chronos2Pipeline:
    return Chronos2Pipeline.from_pretrained(HF_REPO, device_map="cpu")


class _Chronos2Embedder(nn.Module):
    """ONNX-exportable Chronos 2 encoder wrapper.

    Implements the embedding path directly with deterministic static shapes,
    bypassing model.encode() to avoid dynamic-shape ONNX tracing issues.
    All intermediate tensors have shapes that ORT can fully infer statically.
    """

    def __init__(self, pipeline: Chronos2Pipeline, context_length: int) -> None:
        super().__init__()
        model = pipeline.model
        self.patch_size: int = model.chronos_config.input_patch_size
        self.num_patches: int = context_length // self.patch_size
        self.time_encoding_scale: float = float(
            model.chronos_config.time_encoding_scale
        )
        self.use_arcsinh: bool = bool(model.chronos_config.use_arcsinh)
        self.norm_eps: float = 1e-5

        # Model submodules
        self.input_patch_embedding = model.input_patch_embedding
        self.encoder = model.encoder

        # Pre-computed constants (registered as buffers → constant in ONNX)
        N = context_length
        P = self.patch_size
        # Context time encoding: [-N, -(N-1), ..., -1] / time_encoding_scale
        # Shape: (num_patches, patch_size) → unsqueeze(0) in forward for broadcast
        time_enc = (
            torch.arange(-N, 0, dtype=torch.float32)
            .view(self.num_patches, P)
            .div(self.time_encoding_scale)
        )
        self.register_buffer("time_enc", time_enc)  # (num_patches, patch_size)

        # Pre-computed [REG] token embedding; shape (1, 1, d_model)
        reg_id = model.config.reg_token_id
        with torch.no_grad():
            reg_embed = model.shared(torch.tensor([[reg_id]], dtype=torch.long))
        self.register_buffer("reg_embed", reg_embed.detach())  # (1, 1, d_model)

        # Pre-computed future patch input (zeros): shape (1, 1, patch_size * 3)
        P3 = self.patch_size * 3
        self.register_buffer(
            "future_patch_input",
            torch.zeros(1, 1, P3, dtype=torch.float32),
        )

    def _instance_norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked mean/std normalisation + optional arcsinh, ONNX-compatible."""
        # x: (batch, context_length), NaN = padding
        valid = ~torch.isnan(x)
        zeros = torch.zeros_like(x)
        x_safe = torch.where(valid, x, zeros)  # NaN → 0 (NaN * 0.0 ≠ 0 in IEEE 754)

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
        # context: (batch, context_length), float32, NaN = left-padded / missing
        batch = context.shape[0]

        # --- 1. Instance normalisation ---
        normed, valid = self._instance_norm(context)  # (batch, N)

        # --- 2. Patching: reshape to (batch, num_patches, patch_size) ---
        normed = normed.to(
            torch.float16
        )  # match model dtype (bfloat16 on GPU, float16 trace)
        # Note: model was loaded in float32 (device_map=cpu); cast is safe
        normed = normed.to(torch.float32)
        patches = normed.view(batch, self.num_patches, self.patch_size)  # (B, K, P)
        mask_patches = valid.float().view(batch, self.num_patches, self.patch_size)

        # --- 3. Context time encoding: broadcast buffer over batch ---
        time_enc = self.time_enc.unsqueeze(0).expand(batch, -1, -1)  # (B, K, P)

        # Zero out patches that are entirely masked (all observations are NaN)
        patch_valid = mask_patches.sum(dim=-1) > 0  # (B, K)
        patches = patches * patch_valid.float().unsqueeze(-1)

        # --- 4. Concatenate [time | values | mask] → input to patch embedding ---
        patched_context = torch.cat(
            [time_enc, patches, mask_patches], dim=-1
        )  # (B, K, 3P)

        # --- 5. Input patch embedding ---
        input_embeds = self.input_patch_embedding(patched_context)  # (B, K, d_model)

        # --- 6. Append [REG] token ---
        reg = self.reg_embed.expand(batch, -1, -1)  # (B, 1, d_model)
        future_emb = self.input_patch_embedding(
            self.future_patch_input.expand(batch, -1, -1)  # (B, 1, 3P)
        )  # (B, 1, d_model)
        input_embeds = torch.cat(
            [input_embeds, reg, future_emb], dim=1
        )  # (B, K+2, d_model)

        # --- 7. Attention mask: 1 for valid context patches + 1 for [REG] + 1 for future ---
        future_reg_ones = torch.ones(
            batch, 2, dtype=torch.float32, device=context.device
        )
        attention_mask = torch.cat(
            [patch_valid.float(), future_reg_ones], dim=1
        )  # (B, K+2)

        # --- 8. Group IDs: each series in its own group (no cross-series attention) ---
        group_ids = torch.arange(batch, dtype=torch.long, device=context.device)

        # --- 9. Encoder ---
        encoder_out = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
        )
        hidden = encoder_out[0]  # (B, K+2, d_model)

        # --- 10. Pool over context patches only (drop [REG] and future-patch tokens) ---
        sequence = hidden[:, : self.num_patches, :]  # (B, K, d_model)
        patch_valid_f = patch_valid.float().unsqueeze(-1)  # (B, K, 1)
        mean = (sequence * patch_valid_f).sum(dim=1) / patch_valid_f.sum(dim=1).clamp(
            min=1.0
        )

        return mean, sequence


def run_export(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_REPO} ...")
    pipeline = _load_pipeline()
    pipeline.model.eval()

    embedder = _Chronos2Embedder(pipeline, CONTEXT_LENGTH)
    embedder.eval()

    d_model: int = pipeline.model.model_dim
    num_patches = CONTEXT_LENGTH // pipeline.model.chronos_config.input_patch_size

    dummy_context = torch.zeros(2, CONTEXT_LENGTH, dtype=torch.float32)
    # Left-pad first sample with NaN to exercise the masking path
    dummy_context[0, : CONTEXT_LENGTH // 2] = float("nan")

    input_names = ["context"]
    output_names = ["mean", "sequence"]
    dynamic_axes = {
        "context": {0: "batch"},
        "mean": {0: "batch"},
        "sequence": {0: "batch"},
    }

    out_path = output_dir / f"{OUTPUT_PREFIX}.onnx"
    print(f"Exporting {out_path.name}  (context_length={CONTEXT_LENGTH}) ...")

    with torch.no_grad():
        torch.onnx.export(
            embedder,
            (dummy_context,),
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
        print(f"  output '{out.name}': {dims}")
    print(f"  d_model={d_model}, num_patches={num_patches}")
    print("Export complete.")
