"""Compare single-output vs multi-output ONNX files for ATAT.

Usage (from repo root):
    uv run --project models/atat python models/atat/benchmark_multi.py

Produces:
  - models/atat/out/onnx/atat_multi.onnx  (if not already present)
  - File size comparison table
  - onnxruntime latency comparison (CPU)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ONNX_DIR = Path(__file__).resolve().parent / "out" / "onnx"
N_WARMUP = 5
N_RUNS = 50


def _ensure_exports() -> None:
    from atat_prep.export import run_export, run_export_multi

    single_files = [ONNX_DIR / f"atat_{v}.onnx" for v in ("token", "mean", "full")]
    if not all(p.exists() for p in single_files):
        print("Exporting single-output files ...")
        run_export(ONNX_DIR)

    multi_file = ONNX_DIR / "atat_multi.onnx"
    if not multi_file.exists():
        print("Exporting multi-output file ...")
        run_export_multi(ONNX_DIR)


def _make_inputs(batch: int = 4) -> dict:
    from atat_prep.config import DATASET_CHANNEL, SEQ_LEN

    rng = np.random.default_rng(0)
    return {
        "data": rng.standard_normal((batch, SEQ_LEN, DATASET_CHANNEL)).astype(
            np.float32
        ),
        "time": rng.uniform(0, 1, (batch, SEQ_LEN, DATASET_CHANNEL)).astype(np.float32),
        "mask": np.ones((batch, SEQ_LEN, DATASET_CHANNEL), dtype=np.float32),
    }


def _bench(session, feed: dict, n_warmup: int, n_runs: int) -> float:
    """Return mean inference time in milliseconds."""
    for _ in range(n_warmup):
        session.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, feed)
    return (time.perf_counter() - t0) / n_runs * 1000


def main() -> None:
    import onnxruntime as ort

    _ensure_exports()

    inputs = _make_inputs()

    variants = ("token", "mean", "full")
    single_paths = {v: ONNX_DIR / f"atat_{v}.onnx" for v in variants}
    multi_path = ONNX_DIR / "atat_multi.onnx"

    # --- file sizes ---
    print("\n=== File sizes ===")
    total_single = 0
    for v, p in single_paths.items():
        sz = p.stat().st_size / 1024**2
        total_single += p.stat().st_size
        print(f"  atat_{v}.onnx : {sz:.1f} MB")
    print(f"  total (3 files) : {total_single / 1024**2:.1f} MB")
    multi_sz = multi_path.stat().st_size / 1024**2
    print(f"  atat_multi.onnx : {multi_sz:.1f} MB")
    print(f"  ratio multi/total: {multi_path.stat().st_size / total_single:.3f}")

    # --- latency: running all three single files sequentially ---
    print(f"\n=== Latency (CPU, batch=4, {N_RUNS} runs after {N_WARMUP} warmup) ===")
    single_sessions = {v: ort.InferenceSession(str(p)) for v, p in single_paths.items()}
    total_single_ms = 0.0
    for v, sess in single_sessions.items():
        ms = _bench(sess, inputs, N_WARMUP, N_RUNS)
        total_single_ms += ms
        print(f"  atat_{v}.onnx : {ms:.2f} ms")
    print(f"  3 files total   : {total_single_ms:.2f} ms")

    multi_session = ort.InferenceSession(str(multi_path))
    multi_ms = _bench(multi_session, inputs, N_WARMUP, N_RUNS)
    print(f"  atat_multi.onnx : {multi_ms:.2f} ms")
    print(f"  speedup multi vs sequential: {total_single_ms / multi_ms:.2f}x")

    # --- verify outputs match ---
    print("\n=== Numerical consistency ===")
    single_outs = {v: sess.run(None, inputs)[0] for v, sess in single_sessions.items()}
    multi_outs = multi_session.run(None, inputs)
    for (v, single_out), multi_out in zip(single_outs.items(), multi_outs):
        max_diff = np.abs(single_out - multi_out).max()
        print(f"  {v}: max |single - multi| = {max_diff:.2e}")


if __name__ == "__main__":
    main()
