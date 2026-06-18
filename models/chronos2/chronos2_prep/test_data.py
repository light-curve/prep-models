"""Generate test data: synthetic periodic light curves + Chronos 2 embeddings.

No real survey data is required.  We generate sinusoidal light curves with
Gaussian noise and irregular sampling, matching the characteristics of
variable-star observations.

Following the StarEmbed approach, timestamps are discarded and only magnitude
values are passed to Chronos 2 (the model treats observations as sequentially
ordered).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from chronos2_prep.config import CONTEXT_LENGTH
from chronos2_prep.export import _Chronos2Embedder, _load_pipeline

_RNG = np.random.default_rng(42)

# Observation schema stored in the parquet (same as other models in this repo)
_OBS_TYPE = pa.struct(
    [
        pa.field("mjd", pa.float64()),
        pa.field("mag", pa.float32()),
        pa.field("magerr", pa.float32()),
    ]
)


def _synthetic_curve(n_obs: int) -> dict:
    """Return one synthetic periodic light curve as a dict."""
    period = _RNG.uniform(10.0, 200.0)
    amplitude = _RNG.uniform(0.05, 0.5)
    phase = _RNG.uniform(0.0, 2 * np.pi)
    noise_sigma = _RNG.uniform(0.01, 0.05)
    baseline = _RNG.uniform(15.0, 20.0)

    # Irregular time sampling over ~3 years
    mjd = np.sort(_RNG.uniform(58000.0, 59095.0, size=n_obs))
    mag = baseline + amplitude * np.sin(2 * np.pi * mjd / period + phase)
    magerr = _RNG.uniform(noise_sigma * 0.5, noise_sigma * 1.5, size=n_obs).astype(
        np.float32
    )
    mag = (mag + _RNG.normal(0.0, noise_sigma, size=n_obs)).astype(np.float32)

    return {
        "mjd": mjd.tolist(),
        "mag": mag.tolist(),
        "magerr": magerr.tolist(),
        "period": float(period),
        "amplitude": float(amplitude),
    }


def _to_context_tensor(curves: list[dict]) -> torch.Tensor:
    """Left-pad magnitude arrays to CONTEXT_LENGTH, NaN for padding."""
    batch = torch.full((len(curves), CONTEXT_LENGTH), float("nan"), dtype=torch.float32)
    for i, curve in enumerate(curves):
        mag = torch.tensor(curve["mag"], dtype=torch.float32)
        n = min(len(mag), CONTEXT_LENGTH)
        batch[i, CONTEXT_LENGTH - n :] = mag[-n:]
    return batch


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_obs_range = (50, min(CONTEXT_LENGTH, 300))
    print(f"Generating {n_samples} synthetic periodic light curves ...")
    curves = [_synthetic_curve(_RNG.integers(*n_obs_range)) for _ in range(n_samples)]

    context = _to_context_tensor(curves)

    print("Loading Chronos 2 and computing embeddings ...")
    pipeline = _load_pipeline()
    pipeline.model.eval()
    embedder = _Chronos2Embedder(pipeline, CONTEXT_LENGTH)
    embedder.eval()

    with torch.no_grad():
        mean_emb, _sequence = embedder(context)
    embeddings = mean_emb.numpy()  # (n_samples, d_model)

    out_path = output_dir / "chronos2_test_data.parquet"
    _save(curves, embeddings, out_path)
    print(f"Saved {n_samples} samples → {out_path}")


def _save(curves: list[dict], embeddings: np.ndarray, path: Path) -> None:
    d_model = embeddings.shape[1]
    schema = pa.schema(
        [
            pa.field("lightcurve", pa.list_(_OBS_TYPE)),
            pa.field("period", pa.float64()),
            pa.field("amplitude", pa.float64()),
            pa.field("embedding_mean", pa.list_(pa.float32(), d_model)),
        ]
    )
    rows = []
    for i, curve in enumerate(curves):
        obs = [
            {"mjd": float(t), "mag": float(m), "magerr": float(e)}
            for t, m, e in zip(curve["mjd"], curve["mag"], curve["magerr"])
        ]
        rows.append(
            {
                "lightcurve": obs,
                "period": curve["period"],
                "amplitude": curve["amplitude"],
                "embedding_mean": embeddings[i].tolist(),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
