"""Numerical validation: compare monkey-patched ONNX wrappers against the original model.

Run with:
    uv run --project models/atcat python -m atcat_prep validate
"""

from __future__ import annotations

import sys

import torch

from atcat_prep.export import _ATCATEmbedder, _patch_model_for_onnx
from atcat_prep.runtime import add_code_to_path, load_experiment

# The attention replacement (flex_attention → SDPA) introduces small numerical
# differences because the two kernels compute softmax in slightly different orders
# in bfloat16.  This tolerance covers that while still catching logic errors.
_ATTN_ATOL = 5e-2


def _make_dummy_inputs(batch: int = 3, seq_len: int = 243, seed: int = 0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    flux = torch.randn(batch, seq_len, generator=rng)
    flux_err = torch.rand(batch, seq_len, generator=rng).abs() + 0.01
    time = torch.rand(batch, seq_len, generator=rng) * 1000
    # Vary valid sequence lengths to stress masking
    lengths = [50, 150, 243]
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    channel_index = torch.randint(0, 6, (batch, seq_len), generator=rng)
    return flux, flux_err, time, mask, channel_index


def _orig_embedder_forward(
    model_orig, flux, flux_err, time, mask, channel_index, aggregation
):
    """Run the original (unpatched) model and return embeddings the same way _ATCATEmbedder does."""
    ch_wl = torch.tensor([3671.0, 4827.0, 6223.0, 7546.0, 8691.0, 9710.0])[
        channel_index
    ]
    num_lc_points = mask.to(torch.int64).sum(dim=1)

    augmented = model_orig.augment_inputs(num_lc_points)

    from atcat.layers.decoder_model import DecoderInput

    decoder_input = DecoderInput(
        gather_indices=augmented["gather_indices"],
        atat_fine_class_indices=augmented["atat_fine_class_indices"],
        packed_num_elements=augmented["packed_num_elements"],
        flux=flux,
        flux_err=flux_err,
        time=time,
        channel_index=channel_index,
        channel_wavelength=ch_wl,
        mask=mask,
    )

    embedded = model_orig.embed_input(decoder_input)
    last_layer = model_orig.transformer(
        embedded.embeddings,
        mask=embedded.mask,
        time=embedded.time,
    )

    if aggregation == "token":
        cls_idx = embedded.cls_token_index
        return last_layer.gather(
            dim=1,
            index=cls_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, last_layer.shape[-1]),
        ).squeeze(1)
    if aggregation == "mean":
        mask_f = embedded.mask.unsqueeze(-1).to(last_layer.dtype)
        return (last_layer * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    if aggregation == "full":
        return last_layer
    raise ValueError(aggregation)


def run_validate() -> None:
    add_code_to_path()
    from atcat.layers.embed.embed_inputs import LcInputs

    print("Building model ...")
    exp = load_experiment()

    model_orig = exp.architecture.make_model(exp)
    model_orig.eval()
    model_patched = exp.architecture.make_model(exp)
    model_patched.load_state_dict(model_orig.state_dict())
    model_patched.eval()
    _patch_model_for_onnx(model_patched)

    flux, flux_err, time, mask, channel_index = _make_dummy_inputs()
    ch_wl = torch.tensor([3671.0, 4827.0, 6223.0, 7546.0, 8691.0, 9710.0])[
        channel_index
    ]
    inputs = (flux, flux_err, time, mask, channel_index)

    failed = False

    # ── 1. LC embedder: vmap vs batched equivalent ────────────────────────────
    print("\n[1] LC embedder (vmap → batched replacement)")
    lc_inputs = LcInputs(
        flux=flux,
        flux_err=flux_err,
        time=time,
        mask=mask,
        channel_index=channel_index,
        channel_wavelength=ch_wl,
    )
    with torch.no_grad():
        emb_orig = model_orig.lc_embedder(lc_inputs)
        emb_patched = model_patched.lc_embedder(lc_inputs)

    diff = (emb_orig - emb_patched).abs().max().item()
    status = "OK" if diff <= 1e-5 else "FAIL"
    print(f"  Max abs diff: {diff:.2e}  [{status}]")
    if status == "FAIL":
        failed = True

    # ── 2. Full embedder: original flex_attention vs patched SDPA ─────────────
    print(f"\n[2] Full embedder: original vs patched  (atol={_ATTN_ATOL})")
    print("    (flex_attention → SDPA introduces small bfloat16 kernel differences)")
    with torch.no_grad():
        for agg in ("token", "mean", "full"):
            out_orig = _orig_embedder_forward(model_orig, *inputs, agg)
            embedder = _ATCATEmbedder(model_patched, agg)
            embedder.eval()
            out_patched = embedder(*inputs)
            diff = (out_orig.float() - out_patched.float()).abs().max().item()
            status = "OK" if diff <= _ATTN_ATOL else "FAIL"
            print(f"  {agg}: max abs diff = {diff:.2e}  [{status}]")
            if status == "FAIL":
                failed = True

    print()
    if failed:
        print("FAILED", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")
