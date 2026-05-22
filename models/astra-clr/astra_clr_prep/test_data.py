"""Generate test data: synthetic ZTF-like light curves with AstraCLR embeddings.

Preprocessing constants and tensor layout follow the astra-infer reference
implementation (https://github.com/snad-space/astra-infer).
"""

from pathlib import Path

import numpy as np
import onnxruntime as rt

from prep_models_utils.parquet import save_test_data

from astra_clr_prep.config import (
    MODEL_DIR,
    OUTPUT_FILENAME,
    WEIGHTS_DIR,
    HF_ONNX_FILENAME,
)

# ── Preprocessing constants (from astra-infer) ───────────────────────────────

BANDS = ["g", "r", "i"]
SEQ_PER_BAND = {"g": 300, "r": 350, "i": 50}
SEQ_LEN = 700  # 300 + 350 + 50

MJD_OFFSET = 58_000.0

LG_EFF_WAVE = {
    "g": np.log10(4746.48),
    "r": np.log10(6366.38),
    "i": np.log10(7829.03),
}


def _preprocess_lc(lc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (input, times, band_info, mask) tensors of shape (1, SEQ_LEN[, 1])."""
    mjd = np.asarray(lc["mjd"], dtype=np.float64)
    mag = np.asarray(lc["mag"], dtype=np.float32)
    magerr = np.asarray(lc["magerr"], dtype=np.float32)
    band = np.asarray(lc["band"])

    order = np.argsort(mjd)
    mjd, mag, magerr, band = mjd[order], mag[order], magerr[order], band[order]

    norm_mag = np.zeros(SEQ_LEN, dtype=np.float32)
    norm_time = np.zeros(SEQ_LEN, dtype=np.float32)
    band_info = np.zeros(SEQ_LEN, dtype=np.float32)
    mask = np.ones(SEQ_LEN, dtype=np.float32)  # 1 = padded, 0 = real

    offset = 0
    for b in BANDS:
        n = SEQ_PER_BAND[b]
        idx = np.where(band == b)[0][:n]  # "beginning" strategy
        n_real = len(idx)
        if n_real > 0:
            weights = magerr[idx] ** -2
            weighted_mean = np.average(mag[idx], weights=weights)
            norm_mag[offset : offset + n_real] = mag[idx] - weighted_mean
            norm_time[offset : offset + n_real] = mjd[idx] - MJD_OFFSET
            band_info[offset : offset + n_real] = LG_EFF_WAVE[b]
            mask[offset : offset + n_real] = 0.0
        offset += n

    return (
        norm_mag.reshape(1, SEQ_LEN, 1),
        norm_time.reshape(1, SEQ_LEN, 1),
        band_info.reshape(1, SEQ_LEN, 1),
        mask.reshape(1, SEQ_LEN),
    )


def _find_onnx() -> Path:
    candidates = [
        MODEL_DIR / "out" / "onnx" / OUTPUT_FILENAME,
        WEIGHTS_DIR / HF_ONNX_FILENAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "ONNX not found. Run 'prep-models astra-clr download' and 'export' first."
    )


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sess = rt.InferenceSession(str(_find_onnx()))

    rng = np.random.default_rng(42)
    rows = []

    for _ in range(n_samples):
        n_obs = int(rng.integers(200, 500))
        mjd = np.sort(rng.uniform(58000.0, 60500.0, n_obs))
        band = rng.choice(BANDS, n_obs, p=[0.40, 0.50, 0.10])
        mag_base = float(rng.uniform(17.0, 21.0))
        mag = (mag_base + rng.normal(0.0, 0.1, n_obs)).astype(np.float32)
        magerr = rng.uniform(0.01, 0.15, n_obs).astype(np.float32)

        lc = {"mjd": mjd, "mag": mag, "magerr": magerr, "band": band}
        inp, times, binfo, mask = _preprocess_lc(lc)

        (embedding,) = sess.run(
            None,
            {"input": inp, "times": times, "band_info": binfo, "mask": mask},
        )

        rows.append(
            {
                "lightcurve": [
                    {"mjd": float(t), "mag": float(m), "magerr": float(e)}
                    for t, m, e in zip(mjd, mag, magerr)
                ],
                "embedding_mean": embedding[0].tolist(),
            }
        )

    path = output_dir / "astra_clr_test.parquet"
    save_test_data(rows, path)
    print(f"Saved {n_samples} test samples to {path}")
