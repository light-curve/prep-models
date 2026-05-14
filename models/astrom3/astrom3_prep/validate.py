"""Numerical validation: verify the patched model's output shapes and masking behaviour.

Unlike ATCAT (where ProbSparse→SDPA introduces small bfloat16 differences that
can still be compared), for AstroM3 the ProbSparse→SDPA swap is a significant
algorithmic change, so we cannot compare original vs patched outputs.  Instead
we run the patched _AstroM3Embedder and check that:
  1. Outputs have the expected shapes.
  2. No NaN / Inf values are produced.
  3. Masking is respected: zeroing out timesteps changes the mean embedding.

Run with:
    uv run --project models/astrom3 python -m astrom3_prep validate
"""

from __future__ import annotations

import sys

import torch

from astrom3_prep.config import ENC_IN, SEQ_LEN
from astrom3_prep.export import _AstroM3Embedder, load_export_model

_D_MODEL = 128


def run_validate() -> None:
    print("Loading patched model ...")
    model = load_export_model()
    wrapper = _AstroM3Embedder(model)
    wrapper.eval()

    rng = torch.Generator()
    rng.manual_seed(42)
    batch = 3
    x_enc = torch.randn(batch, SEQ_LEN, ENC_IN, generator=rng)
    mask_full = torch.ones(batch, SEQ_LEN)
    mask_partial = mask_full.clone()
    mask_partial[0, 150:] = 0.0
    mask_partial[1, 80:] = 0.0

    failed = False

    with torch.no_grad():
        mean_out, sequence_out = wrapper(x_enc, mask_full)

    # [1] Shape checks
    print("\n[1] Output shapes")
    for name, tensor, expected in [
        ("mean", mean_out, (batch, _D_MODEL)),
        ("sequence", sequence_out, (batch, SEQ_LEN, _D_MODEL)),
    ]:
        ok = tuple(tensor.shape) == expected
        status = (
            "OK" if ok else f"FAIL (got {tuple(tensor.shape)}, expected {expected})"
        )
        print(f"  {name}: {status}")
        if not ok:
            failed = True

    # [2] No NaN / Inf
    print("\n[2] NaN / Inf check")
    for name, tensor in [("mean", mean_out), ("sequence", sequence_out)]:
        ok = bool(torch.isfinite(tensor).all().item())
        print(f"  {name}: {'OK' if ok else 'FAIL (contains NaN/Inf)'}")
        if not ok:
            failed = True

    # [3] Masking changes the mean embedding
    print("\n[3] Masking affects mean output")
    with torch.no_grad():
        mean_partial, _ = wrapper(x_enc, mask_partial)
    diff = (mean_out - mean_partial).abs().max().item()
    ok = diff > 0
    print(
        f"  Max abs diff (full vs partial mask): {diff:.4f}  [{'OK' if ok else 'FAIL'}]"
    )
    if not ok:
        failed = True

    print()
    if failed:
        print("FAILED", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")
