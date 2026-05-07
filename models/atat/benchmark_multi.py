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
BATCH_SIZE = 32


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


def _bench_output(
    session, feed: dict, output_name: str, n_warmup: int, n_runs: int
) -> float:
    """Return mean inference time in milliseconds, requesting a single named output."""
    for _ in range(n_warmup):
        session.run([output_name], feed)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run([output_name], feed)
    return (time.perf_counter() - t0) / n_runs * 1000


def main() -> None:
    import onnxruntime as ort

    _ensure_exports()

    inputs = _make_inputs(batch=BATCH_SIZE)

    variants = ("token", "mean", "full")
    single_paths = {v: ONNX_DIR / f"atat_{v}.onnx" for v in variants}
    multi_path = ONNX_DIR / "atat_multi.onnx"

    single_sessions = {v: ort.InferenceSession(str(p)) for v, p in single_paths.items()}
    multi_session = ort.InferenceSession(str(multi_path))

    # --- file sizes ---
    print("\n=== File sizes ===")
    multi_sz_mb = multi_path.stat().st_size / 1024**2
    print(f"  atat_multi.onnx : {multi_sz_mb:.1f} MB  (shared across all variants)")
    for v, p in single_paths.items():
        print(f"  atat_{v}.onnx   : {p.stat().st_size / 1024**2:.1f} MB")

    # --- per-variant latency comparison ---
    print(
        f"\n=== Latency per variant (CPU, batch={BATCH_SIZE}, {N_RUNS} runs after {N_WARMUP} warmup) ==="
    )
    print(f"  {'variant':<8}  {'single (ms)':>12}  {'multi (ms)':>11}  {'ratio':>7}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 11}  {'-' * 7}")
    for v, sess in single_sessions.items():
        single_ms = _bench(sess, inputs, N_WARMUP, N_RUNS)
        multi_ms = _bench_output(
            multi_session, inputs, f"embedding_{v}", N_WARMUP, N_RUNS
        )
        print(
            f"  {v:<8}  {single_ms:>12.2f}  {multi_ms:>11.2f}  {multi_ms / single_ms:>7.2f}x"
        )

    # --- verify outputs match ---
    print("\n=== Numerical consistency ===")
    for v, sess in single_sessions.items():
        single_out = sess.run(None, inputs)[0]
        (multi_out,) = multi_session.run([f"embedding_{v}"], inputs)
        max_diff = np.abs(single_out - multi_out).max()
        print(f"  {v}: max |single - multi| = {max_diff:.2e}")


if __name__ == "__main__":
    main()
