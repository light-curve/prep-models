"""Export ATCAT LC-only transformer embeddings to ONNX."""

from __future__ import annotations

import types
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from torch import nn

from atcat_prep.config import OUTPUT_PREFIX
from atcat_prep.runtime import add_code_to_path, load_model


def _patch_model_for_onnx(model: nn.Module) -> None:
    """Monkey-patch upstream model internals that block ONNX tracing.

    Two issues prevent clean ONNX export:
    1. `torch.func.vmap` in PositionalEncoderWithChannelTransform.intermediate_outputs
       -- not supported by the TorchScript-based ONNX tracer.
    2. `flex_attention` in DecoderSelfAttention/LocalDecoderAttention.forward
       -- not exportable to ONNX; replaced with SDPA.
    """
    _patch_lc_embedder(model.lc_embedder)
    for layer in model.transformer.layers:
        _patch_attn_layer(layer.attn)


def _patch_lc_embedder(lc_embedder: nn.Module) -> None:
    # PositionalEncoderWithChannelTransform.intermediate_outputs uses
    # torch.func.vmap to vectorise the per-sample positional encoding.
    # vmap is not supported by PyTorch's TorchScript-based ONNX tracer
    # (torch.onnx.export with dynamo=False), so we replace it with an
    # equivalent batched implementation that the tracer can handle.
    from atcat.layers.embed import lc_embed

    def _intermediate_outputs_no_vmap(self, inputs):
        base = self.base_encoder
        time = inputs.time  # [batch, seq_len]
        power = lc_embed.piecewise_linear(base.exp_input, base.y0, base.y1, base.y2)
        inv_period = torch.pow(
            torch.tensor(
                1.0 / (10.0 * base.T_max), device=time.device, dtype=time.dtype
            ),
            power.to(time.dtype),
        )  # [emb_dim//2]
        arg = time.unsqueeze(-1) * inv_period.unsqueeze(0).unsqueeze(0)
        sin_t = torch.sin(arg)
        cos_t = torch.cos(arg)
        b, s = time.shape
        sin_cos_t = torch.stack((sin_t, cos_t), dim=-1).view(b, s, base.emb_dim)
        sin_cos_t = sin_cos_t / 10.0
        time_embed = self.time_base_transform(sin_cos_t)

        flux_embed = self.flux_emb(inputs.flux)
        flux_err_embed = (
            self.flux_err_emb(inputs.flux_err)
            if self.flux_err_emb is not None
            else None
        )

        from atcat.elasticc import lsst_colors
        from atcat.layers import rotary_encoder

        rotation_amount = (
            5.0 * lsst_colors.norm_lsst_wavelength(inputs.channel_wavelength)
            if self.use_wavelengths
            else inputs.channel_index.to(time_embed.dtype)
        )
        time_embed = rotary_encoder.embed_matrix_fast(
            x_batched=time_embed,
            m_batched=rotation_amount,
            thetas=self.channel_emb_thetas,
        )

        if not self.use_wavelengths:
            batch_size = inputs.flux.shape[0]
            ch_embed = torch.gather(
                self.channel_emb.unsqueeze(0).expand(batch_size, -1, -1),
                dim=1,
                index=inputs.channel_index.unsqueeze(2).expand(-1, -1, self.emb_dim),
            )
            time_embed = time_embed + ch_embed

        flux_embed = self.flux_dropout(flux_embed)
        flux_err_embed = (
            self.flux_err_dropout(flux_err_embed)
            if flux_err_embed is not None
            else None
        )

        return lc_embed.IntermediateEmbedOutputs(
            flux_embed=flux_embed,
            flux_err_embed=flux_err_embed,
            time_embed=time_embed,
            flux_emb_weight=self.flux_emb_weight,
            flux_err_emb_weight=self.flux_err_emb_weight
            if flux_err_embed is not None
            else None,
            time_emb_weight=self.time_emb_weight,
        )

    lc_embedder.intermediate_outputs = types.MethodType(
        _intermediate_outputs_no_vmap, lc_embedder
    )


def _patch_attn_layer(attn: nn.Module) -> None:
    """Replace flex_attention forward with ONNX-compatible SDPA.

    Handles two upstream attention types:
    - DecoderSelfAttention: causal mask only, forward(x)
    - LocalDecoderAttention: causal + time-local mask, forward(x, time)

    flex_attention is not exportable to ONNX: the tracer cannot capture its
    score_mod/mask_mod callables.  We replace it with SDPA with a boolean mask.
    """
    from atcat.layers import flex_mha

    if isinstance(attn, flex_mha.DecoderSelfAttention):

        def _forward_causal(self, x: torch.Tensor) -> torch.Tensor:
            original_dtype = x.dtype
            if self.dtype is not None:
                x = x.to(self.dtype)
            batch_size, seq_len, _ = x.shape
            q, k, v = self.get_qkv(x)
            idx = torch.arange(seq_len, device=x.device)
            causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)  # [seq_len, seq_len]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
            out = out.permute(0, 2, 1, 3).reshape(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            result = self.out_emb(out) if self.out_emb is not None else out
            return result.to(original_dtype)

        attn.forward = types.MethodType(_forward_causal, attn)

    elif isinstance(attn, flex_mha.LocalDecoderAttention):

        def _forward_local(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            original_dtype = x.dtype
            if self.dtype is not None:
                x = x.to(self.dtype)
            batch_size, seq_len, _ = x.shape
            q, k, v = self.get_qkv(x)
            idx = torch.arange(seq_len, device=x.device)
            causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # [seq_len, seq_len]
            time_diff = (
                time.unsqueeze(2) - time.unsqueeze(1)
            ).abs()  # [batch, seq_len, seq_len]
            local = time_diff <= self.time_threshold  # [batch, seq_len, seq_len]
            attn_mask = (causal.unsqueeze(0) & local).unsqueeze(
                1
            )  # [batch, 1, seq_len, seq_len]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out = out.permute(0, 2, 1, 3).reshape(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            result = self.out_emb(out) if self.out_emb is not None else out
            return result.to(original_dtype)

        attn.forward = types.MethodType(_forward_local, attn)

    else:
        raise TypeError(
            f"Unexpected attention type for ONNX patch: {type(attn).__name__}"
        )


class _ATCATEmbedder(nn.Module):
    """ONNX wrapper returning last, mean, and sequence embeddings in one forward pass.

    Bypasses embed_input() and schema.embed() (which contain torch._assert calls
    incompatible with ONNX tracing) by driving the lc_embedder and transformer
    directly.  Only valid for the LC-only checkpoint (add_predict_token=False,
    no metadata embedder).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.embed_dim: int = model.embed_dim
        self.register_buffer(
            "channel_wavelengths",
            torch.tensor([3671.0, 4827.0, 6223.0, 7546.0, 8691.0, 9710.0]),
            persistent=False,
        )

    def forward(
        self,
        flux: torch.Tensor,
        flux_err: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor,
        channel_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from atcat.layers.embed.embed_inputs import LcInputs

        channel_wavelength = self.channel_wavelengths[channel_index]
        num_lc_points = mask.to(torch.int64).sum(dim=1)
        seq_len = flux.shape[1]

        lc_inputs = LcInputs(
            flux=flux,
            flux_err=flux_err,
            time=time,
            mask=mask,
            channel_index=channel_index,
            channel_wavelength=channel_wavelength,
        )
        lc_emb = self.model.lc_embedder(lc_inputs)

        idx = torch.arange(seq_len, device=flux.device)
        attn_mask = idx.unsqueeze(0) < num_lc_points.unsqueeze(1)

        last_layer = self.model.transformer(lc_emb, mask=attn_mask, time=time)

        cls_idx = (num_lc_points - 1).clamp(min=0)
        embedding_last = last_layer.gather(
            dim=1,
            index=cls_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.embed_dim),
        ).squeeze(1)

        mask_f = attn_mask.unsqueeze(-1).to(last_layer.dtype)
        embedding_mean = (last_layer * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(
            min=1.0
        )

        embedding_sequence = last_layer

        return embedding_last, embedding_mean, embedding_sequence


def load_export_model() -> nn.Module:
    add_code_to_path()
    model = load_model()
    _patch_model_for_onnx(model)
    return model


def run_export(output_dir: Path) -> None:
    from atcat_prep.validate import run_validate

    print("=== Patch validation ===")
    run_validate()
    print("========================\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_export_model()

    batch = 2
    seq_len = 243
    dummy = (
        torch.zeros(batch, seq_len, dtype=torch.float32),
        torch.ones(batch, seq_len, dtype=torch.float32),
        torch.zeros(batch, seq_len, dtype=torch.float32),
        torch.ones(batch, seq_len, dtype=torch.bool),
        torch.zeros(batch, seq_len, dtype=torch.int64),
    )
    input_names = ["flux", "flux_err", "time", "mask", "channel_index"]
    output_names = ["last", "mean", "sequence"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    wrapper = _ATCATEmbedder(model)
    wrapper.eval()
    out_path = output_dir / f"{OUTPUT_PREFIX}_bf16.onnx"
    print(f"Exporting {out_path.name} ...")
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
    print(f"  Saved: {out_path}")
    proto = onnx.load(str(out_path))
    for out in proto.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output '{out.name}': {dims}")
    print("Export complete.")
