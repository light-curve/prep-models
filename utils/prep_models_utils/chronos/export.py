"""Shared ONNX export boilerplate for Chronos-family encoder wrappers."""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn


def run_export(
    output_dir: Path,
    *,
    embedder: nn.Module,
    output_prefix: str,
    d_model: int,
    patch_size: int,
) -> None:
    """Trace ``embedder`` to ``<output_prefix>.onnx`` with a dynamic seq axis.

    The wrapper must accept a single ``context: [batch, seq]`` float32 tensor
    (NaN-padded) and return ``(mean, sequence)``.  ``seq`` must be a multiple of
    ``patch_size``; it is exported as a dynamic axis, so the trace length below
    is only a concrete example.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder.eval()

    trace_len = 32 * patch_size
    dummy_context = torch.zeros(2, trace_len, dtype=torch.float32)
    # Left-pad first sample with NaN to exercise the masking path
    dummy_context[0, : trace_len // 2] = float("nan")

    dynamic_axes = {
        "context": {0: "batch", 1: "seq"},
        "mean": {0: "batch"},
        "sequence": {0: "batch", 1: "num_patches"},
    }

    out_path = output_dir / f"{output_prefix}.onnx"
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
    print(f"  d_model={d_model}, patch_size={patch_size}")
    print("Export complete.")
