"""Generate test data: synthetic two-band supernova light curves + Maven embeddings.

Maven was fine-tuned on ZTF Bright Transient Survey light curves, which are
only distributed through the upstream HuggingFace dataset.  To keep this
command self-contained we instead simulate ZTF-like R/g light curves from a
Bazin (2009) flux model with irregular cadence, seasonal gaps and photometric
noise, then apply the upstream preprocessing (`src.dataloader.load_lightcurves`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from maven_prep.config import BANDS, N_MAX_OBS, SEQ_LEN, TEST_DATA_FILENAME
from maven_prep.export import MavenEmbedder
from maven_prep.runtime import load_model

_OBS_TYPE = pa.struct(
    [
        pa.field("mjd", pa.float64()),
        pa.field("mag", pa.float32()),
        pa.field("magerr", pa.float32()),
        pa.field("band", pa.string()),
    ]
)

_MIN_OBS_PER_BAND = 5
# More than N_MAX_OBS so that the random-subsampling branch is exercised too.
_MAX_OBS_PER_BAND = 150


def _synthetic_curve(rng: np.random.Generator) -> list[dict]:
    """Return one synthetic two-band supernova light curve as a list of observations."""
    t0 = rng.uniform(58300.0, 59500.0)
    peak_mag = rng.uniform(16.5, 19.0)
    t_rise = rng.uniform(2.0, 8.0)
    t_fall = rng.uniform(15.0, 60.0)
    # g declines faster and is slightly brighter at peak for a young SN.
    band_params = {
        "R": {"dmag": 0.0, "fall": 1.0},
        "g": {"dmag": rng.uniform(-0.3, 0.2), "fall": rng.uniform(0.5, 0.9)},
    }

    obs = []
    for band in BANDS:
        n = int(rng.integers(_MIN_OBS_PER_BAND, _MAX_OBS_PER_BAND + 1))
        mjd = np.sort(rng.uniform(t0 - 30.0, t0 + 150.0, size=n))
        p = band_params[band]
        dt = mjd - t0
        flux = np.exp(-dt / (t_fall * p["fall"])) / (1.0 + np.exp(-dt / t_rise))
        flux /= flux.max()
        # Clip to a ~21 mag limiting depth so faint tails stay finite.
        mag_true = np.minimum(peak_mag + p["dmag"] - 2.5 * np.log10(flux + 1e-3), 21.5)
        magerr = 0.02 + 0.2 * np.exp(mag_true - 21.0)
        mag = mag_true + rng.normal(0.0, magerr)
        obs.extend(
            {"mjd": float(t), "mag": float(m), "magerr": float(e), "band": band}
            for t, m, e in zip(mjd, mag, magerr)
        )
    return obs


def preprocess(
    obs: list[dict], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upstream `load_lightcurves` preprocessing for one light curve.

    Per band (R first, then g): keep at most N_MAX_OBS observations (random
    subsample without replacement if longer), subtract the band's first
    observation time, zero-pad to N_MAX_OBS and build a validity mask.
    Milky Way extinction correction is omitted (synthetic data has none).
    """
    mags, times, masks = [], [], []
    for band in BANDS:
        rows = [o for o in obs if o["band"] == band]
        t = np.array([o["mjd"] for o in rows], dtype=np.float64)
        m = np.array([o["mag"] for o in rows], dtype=np.float64)
        if len(rows) > N_MAX_OBS:
            idx = np.sort(rng.choice(len(rows), N_MAX_OBS, replace=False))
            t, m = t[idx], m[idx]
        n = len(t)
        if n:
            t = t - t.min()
        pad = N_MAX_OBS - n
        times.append(np.pad(t, (0, pad)))
        mags.append(np.pad(m, (0, pad)))
        masks.append(np.pad(np.ones(n), (0, pad)))
    return (
        np.concatenate(mags).astype(np.float32),
        np.concatenate(times).astype(np.float32),
        np.concatenate(masks).astype(np.float32),
    )


def _save(
    curves: list[list[dict]],
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    mean: np.ndarray,
    clip: np.ndarray,
    path: Path,
) -> None:
    mag, time, mask = inputs
    schema = pa.schema(
        [
            pa.field("lightcurve", pa.list_(_OBS_TYPE)),
            pa.field("input_mag", pa.list_(pa.float32(), SEQ_LEN)),
            pa.field("input_time", pa.list_(pa.float32(), SEQ_LEN)),
            pa.field("input_mask", pa.list_(pa.float32(), SEQ_LEN)),
            pa.field("embedding_mean", pa.list_(pa.float32(), mean.shape[1])),
            pa.field("embedding_clip", pa.list_(pa.float32(), clip.shape[1])),
        ]
    )
    rows = [
        {
            "lightcurve": curve,
            "input_mag": mag[i].tolist(),
            "input_time": time[i].tolist(),
            "input_mask": mask[i].tolist(),
            "embedding_mean": mean[i].tolist(),
            "embedding_clip": clip[i].tolist(),
        }
        for i, curve in enumerate(curves)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def run_test_data(output_dir: Path, n_samples: int = 10, *, seed: int = 42) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Generating {n_samples} synthetic two-band supernova light curves ...")
    curves = [_synthetic_curve(rng) for _ in range(n_samples)]
    prepped = [preprocess(c, rng) for c in curves]
    inputs = tuple(np.stack(arrays) for arrays in zip(*prepped))

    embedder = MavenEmbedder(load_model())
    embedder.eval()
    with torch.no_grad():
        mean, clip, _ = embedder(*(torch.from_numpy(a) for a in inputs))

    out_path = output_dir / TEST_DATA_FILENAME
    _save(curves, inputs, mean.numpy(), clip.numpy(), out_path)
    print(f"Saved {n_samples} samples → {out_path}")
