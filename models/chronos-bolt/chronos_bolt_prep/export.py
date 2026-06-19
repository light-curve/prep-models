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

import torch
import torch.nn as nn

from chronos import ChronosBoltPipeline

from prep_models_utils.chronos import masked_instance_norm
from prep_models_utils.chronos import run_export as run_shared_export

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

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # context: (batch, seq), float32, NaN = left-padded / missing.
        # seq must be a multiple of patch_size; it is a dynamic ONNX axis.
        batch = context.shape[0]
        P = self.patch_size

        # --- 1. Instance normalisation (masked positions become 0) ---
        normed, valid = masked_instance_norm(
            context, eps=self.norm_eps, use_arcsinh=self.use_arcsinh
        )  # (batch, seq)

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
    print(f"Loading {hf_repo(size)} ...")
    pipeline = _load_pipeline(size)
    pipeline.model.eval()

    embedder = _ChronosBoltEmbedder(pipeline)
    run_shared_export(
        output_dir,
        embedder=embedder,
        output_prefix=output_prefix(size),
        d_model=pipeline.model.model_dim,
        patch_size=embedder.patch_size,
    )
