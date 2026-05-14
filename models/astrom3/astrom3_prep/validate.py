"""Numerical validation: compare patched PyTorch model against the exported ONNX.

Unlike ATCAT (where ProbSparse→SDPA introduces small bfloat16 kernel differences),
the ProbSparse→SDPA replacement here is a significant algorithmic change, so we
do not compare original vs patched.  Instead we verify that torch.onnx.export
faithfully serialised the patched model by running both and checking agreement.

Run with:
    uv run --project models/astrom3 python -m astrom3_prep validate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from astrom3_prep.config import ENC_IN, OUTPUT_PREFIX, SEQ_LEN
from astrom3_prep.export import _AstroM3Embedder, load_export_model

_DEFAULT_ONNX = (
    Path(__file__).resolve().parents[1] / "out" / "onnx" / f"{OUTPUT_PREFIX}.onnx"
)
_ATOL = 1e-4


def run_validate(onnx_path: Path = _DEFAULT_ONNX) -> None:
    import onnxruntime as rt

    if not onnx_path.exists():
        print(
            f"ONNX file not found at {onnx_path}. Run 'prep-models astrom3 export' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading patched model ...")
    model = load_export_model()
    wrapper = _AstroM3Embedder(model)
    wrapper.eval()

    rng = torch.Generator()
    rng.manual_seed(42)
    batch = 3
    x_enc = torch.randn(batch, SEQ_LEN, ENC_IN, generator=rng)
    mask = torch.ones(batch, SEQ_LEN)
    mask[0, 150:] = 0.0  # vary valid lengths to stress masking
    mask[1, 80:] = 0.0

    with torch.no_grad():
        pt_mean, pt_sequence = wrapper(x_enc, mask)

    sess = rt.InferenceSession(str(onnx_path))
    feeds = {"x_enc": x_enc.numpy(), "mask": mask.numpy()}
    onnx_outputs = sess.run(None, feeds)
    onnx_mean = onnx_outputs[0]
    onnx_sequence = onnx_outputs[1]

    failed = False
    for name, pt_out, onnx_out in [
        ("mean", pt_mean.numpy(), onnx_mean),
        ("sequence", pt_sequence.numpy(), onnx_sequence),
    ]:
        diff = np.abs(pt_out - onnx_out).max()
        status = "OK" if diff <= _ATOL else "FAIL"
        print(f"  {name}: max abs diff = {diff:.2e}  [{status}]")
        if status == "FAIL":
            failed = True

    print()
    if failed:
        print("FAILED", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")
