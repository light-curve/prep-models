"""Generate test data: synthetic periodic light curves + MOMENT embeddings.

No real survey data is required.  We generate sinusoidal light curves with
Gaussian noise and irregular sampling, matching the characteristics of
variable-star observations.

MOMENT takes no timestamp input, so timestamps are discarded and only magnitude
values are passed (same convention this repo uses for Chronos).  MOMENT has a
fixed 512-step context: each curve is capped to its last 512 observations and
left-padded with NaN, exercising the masking path while keeping the sequence
length fixed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from moment_prep.config import SEQ_LEN, output_prefix
from moment_prep.export import _load_model, _MomentEmbedder

# Observation schema stored in the parquet.  The model consumes only the
# magnitudes; mjd is kept to record the (irregular) sampling of each curve.
_OBS_TYPE = pa.struct(
    [
        pa.field("mjd", pa.float64()),
        pa.field("mag", pa.float32()),
    ]
)

_MIN_OBS = 50


def _synthetic_curve(rng: np.random.Generator, n_obs: int) -> dict:
    """Return one synthetic periodic light curve as {mjd, mag}."""
    period = rng.uniform(10.0, 200.0)
    amplitude = rng.uniform(0.05, 0.5)
    phase = rng.uniform(0.0, 2 * np.pi)
    noise_sigma = rng.uniform(0.01, 0.05)
    baseline = rng.uniform(15.0, 20.0)

    # Irregular time sampling over ~3 years
    mjd = np.sort(rng.uniform(58000.0, 59095.0, size=n_obs))
    mag = baseline + amplitude * np.sin(2 * np.pi * mjd / period + phase)
    mag = (mag + rng.normal(0.0, noise_sigma, size=n_obs)).astype(np.float32)

    return {"mjd": mjd.tolist(), "mag": mag.tolist()}


def _log_uniform_n_obs(rng: np.random.Generator, max_obs: int) -> int:
    """Draw a log-uniform observation count in [_MIN_OBS, max_obs]."""
    log_n = rng.uniform(np.log(_MIN_OBS), np.log(max_obs))
    return round(float(np.exp(log_n)))


def _context_for_curve(curve: dict) -> torch.Tensor:
    """Left-pad one curve's last SEQ_LEN magnitudes with NaN.

    Returns a (1, SEQ_LEN) tensor; padding is NaN, marking missing observations.
    """
    mag = torch.tensor(curve["mag"][-SEQ_LEN:], dtype=torch.float32)
    n = mag.shape[0]
    out = torch.full((1, SEQ_LEN), float("nan"), dtype=torch.float32)
    out[0, SEQ_LEN - n :] = mag
    return out


def _save(curves: list[dict], embeddings: np.ndarray, path: Path) -> None:
    d_model = embeddings.shape[1]
    schema = pa.schema(
        [
            pa.field("lightcurve", pa.list_(_OBS_TYPE)),
            pa.field("embedding_mean", pa.list_(pa.float32(), d_model)),
        ]
    )
    rows = []
    for i, curve in enumerate(curves):
        obs = [
            {"mjd": float(t), "mag": float(m)}
            for t, m in zip(curve["mjd"], curve["mag"])
        ]
        rows.append(
            {
                "lightcurve": obs,
                "embedding_mean": embeddings[i].tolist(),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def run_test_data(
    output_dir: Path, n_samples: int = 10, *, size: str, seed: int = 42
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Generating {n_samples} synthetic periodic light curves ...")
    # Cap at SEQ_LEN; log-uniform spreads counts across orders of magnitude.
    curves = [
        _synthetic_curve(rng, _log_uniform_n_obs(rng, SEQ_LEN))
        for _ in range(n_samples)
    ]
    sizes = sorted(len(c["mag"]) for c in curves)
    print(f"  observation counts (log-uniform {_MIN_OBS}–{SEQ_LEN}): {sizes}")

    model = _load_model(size)
    embedder = _MomentEmbedder(model)
    embedder.eval()

    d_model = embedder.d_model
    embeddings = np.empty((len(curves), d_model), dtype=np.float32)
    with torch.no_grad():
        for i, curve in enumerate(curves):
            mean_emb, _sequence = embedder(_context_for_curve(curve))
            embeddings[i] = mean_emb.numpy()[0]

    out_path = output_dir / f"{output_prefix(size)}_test.parquet"
    _save(curves, embeddings, out_path)
    print(f"Saved {n_samples} samples → {out_path}")
