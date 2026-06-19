"""Shared synthetic-light-curve test-data generation for Chronos-family models.

No real survey data is required.  We generate sinusoidal light curves with
Gaussian noise and irregular sampling, matching the characteristics of
variable-star observations.

Following the StarEmbed approach, timestamps are discarded and only magnitude
values are passed to the model (it treats observations as sequentially ordered).
Each curve is embedded at its own length (padded to the next multiple of the
patch size), exercising the dynamic ONNX sequence axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

# Observation schema stored in the parquet. The model consumes only the
# magnitudes; mjd is kept to record the (irregular) sampling of each curve.
_OBS_TYPE = pa.struct(
    [
        pa.field("mjd", pa.float64()),
        pa.field("mag", pa.float32()),
    ]
)

_MIN_OBS = 50
_PATCH_SIZE = 16


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
    """Draw a log-uniform observation count in [_MIN_OBS, max_obs].

    Log-uniform spreads samples across orders of magnitude and yields arbitrary
    (odd, non-round) lengths, exercising the dynamic ONNX sequence axis.
    """
    log_n = rng.uniform(np.log(_MIN_OBS), np.log(max_obs))
    return int(round(float(np.exp(log_n))))


def _context_for_curve(curve: dict, max_obs: int) -> torch.Tensor:
    """Left-pad one curve's magnitudes to the next multiple of the patch size.

    Returns a (1, seq) tensor with NaN padding, where seq % _PATCH_SIZE == 0.
    """
    mag = torch.tensor(curve["mag"][-max_obs:], dtype=torch.float32)
    n = mag.shape[0]
    seq = ((n + _PATCH_SIZE - 1) // _PATCH_SIZE) * _PATCH_SIZE
    out = torch.full((1, seq), float("nan"), dtype=torch.float32)
    out[0, seq - n :] = mag
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
    output_dir: Path,
    n_samples: int,
    *,
    embedder: nn.Module,
    d_model: int,
    output_prefix: str,
    max_obs: int,
    seed: int = 42,
) -> None:
    """Generate synthetic curves, embed each with ``embedder``, write parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Generating {n_samples} synthetic periodic light curves ...")
    curves = [
        _synthetic_curve(rng, _log_uniform_n_obs(rng, max_obs))
        for _ in range(n_samples)
    ]
    sizes = sorted(len(c["mag"]) for c in curves)
    print(f"  observation counts (log-uniform {_MIN_OBS}–{max_obs}): {sizes}")

    embedder.eval()
    embeddings = np.empty((len(curves), d_model), dtype=np.float32)
    with torch.no_grad():
        for i, curve in enumerate(curves):
            mean_emb, _sequence = embedder(_context_for_curve(curve, max_obs))
            embeddings[i] = mean_emb.numpy()[0]

    out_path = output_dir / f"{output_prefix}_test.parquet"
    _save(curves, embeddings, out_path)
    print(f"Saved {n_samples} samples → {out_path}")
