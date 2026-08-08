"""Export Chronos 2 encoder to ONNX.

The ONNX model takes a batch of univariate time series (magnitudes, NaN for
padding) and returns encoder patch embeddings.

Following the StarEmbed benchmarking approach, timestamps are not used: the
model treats observations as sequentially ordered, so the preprocessing step
is just magnitude normalization (handled internally by Chronos 2's instance
normalization).

Input
-----
context : float32 [batch, seq]
    Magnitude values.  NaN marks left-padded / missing observations.
    Both axes are dynamic; ``seq`` must be a multiple of the patch size (16),
    up to the model's native context of 8192.

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

import torch
from chronos import Chronos2Pipeline
from prep_models_utils.chronos import masked_instance_norm
from prep_models_utils.chronos import run_export as run_shared_export
from torch import nn

from chronos2_prep.config import HF_REPO, HF_REVISION, OUTPUT_PREFIX


def _load_pipeline() -> Chronos2Pipeline:
    # Pin the HF revision so the loaded weights are 100% reproducible.
    return Chronos2Pipeline.from_pretrained(
        HF_REPO, revision=HF_REVISION, device_map="cpu"
    )


class _Chronos2Embedder(nn.Module):
    """ONNX-exportable Chronos 2 encoder wrapper.

    Implements the embedding path directly with deterministic static shapes,
    bypassing model.encode() to avoid dynamic-shape ONNX tracing issues.
    All intermediate tensors have shapes that ORT can fully infer statically.
    """

    def __init__(self, pipeline: Chronos2Pipeline) -> None:
        super().__init__()
        model = pipeline.model
        self.patch_size: int = model.chronos_config.input_patch_size
        self.time_encoding_scale: float = float(
            model.chronos_config.time_encoding_scale
        )
        self.use_arcsinh: bool = bool(model.chronos_config.use_arcsinh)
        self.norm_eps: float = 1e-5

        # Model submodules
        self.input_patch_embedding = model.input_patch_embedding
        self.encoder = model.encoder

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

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # context: (batch, seq), float32, NaN = left-padded / missing.
        # seq must be a multiple of patch_size; it is a dynamic ONNX axis.
        batch = context.shape[0]
        P = self.patch_size

        # --- 1. Instance normalisation ---
        normed, valid = masked_instance_norm(
            context, eps=self.norm_eps, use_arcsinh=self.use_arcsinh
        )  # (batch, seq)

        # --- 2. Context time encoding: [-seq, ..., -1] / scale ---
        # Built from cumsum-of-ones so the length stays a *dynamic* ONNX axis
        # (torch.arange(-seq, 0) would fold the traced int and pin the length).
        ones = torch.ones_like(context)  # (batch, seq)
        positions = ones.cumsum(dim=1)  # 1, 2, ..., seq  (dynamic)
        length = positions[:, -1:]  # = seq, as a tensor
        time_flat = (positions - length - 1.0) / self.time_encoding_scale  # (B, seq)

        # --- 3. Patching: reshape to (batch, num_patches, patch_size) ---
        patches = normed.view(batch, -1, P)  # (B, K, P), K dynamic
        mask_patches = valid.float().view(batch, -1, P)
        time_enc = time_flat.view(batch, -1, P)  # (B, K, P)

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
        sequence = hidden[:, :-2, :]  # (B, K, d_model); last 2 are [REG] + future
        patch_valid_f = patch_valid.float().unsqueeze(-1)  # (B, K, 1)
        mean = (sequence * patch_valid_f).sum(dim=1) / patch_valid_f.sum(dim=1).clamp(
            min=1.0
        )

        return mean, sequence


def run_export(output_dir: Path) -> None:
    print(f"Loading {HF_REPO} ...")
    pipeline = _load_pipeline()
    pipeline.model.eval()

    embedder = _Chronos2Embedder(pipeline)
    run_shared_export(
        output_dir,
        embedder=embedder,
        output_prefix=OUTPUT_PREFIX,
        d_model=pipeline.model.model_dim,
        patch_size=embedder.patch_size,
    )
