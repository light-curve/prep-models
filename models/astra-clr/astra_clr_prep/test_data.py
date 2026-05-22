"""Generate test data: real ZTF light curves with AstraCLR embeddings.

Reads objects from the snad-space/astra-zubercaldr16_gaiadr3vclassre dataset
(Zubercal DR16 × Gaia DR3). Preprocessing follows the astra-infer reference
implementation (https://github.com/snad-space/astra-infer).
"""

from pathlib import Path

import numpy as np
import onnxruntime as rt
import pyarrow.parquet as pq

from prep_models_utils.parquet import save_test_data

from astra_clr_prep.config import (
    DATA_FILE,
    MODEL_DIR,
    OUTPUT_FILENAME,
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

FILTERID_TO_BAND = {1: "g", 2: "r", 3: "i"}


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
    path = MODEL_DIR / "out" / "onnx" / OUTPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'prep-models astra-clr export' first."
        )
    return path


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run 'prep-models astra-clr download' first."
        )

    sess = rt.InferenceSession(str(_find_onnx()))

    table = pq.read_table(DATA_FILE, columns=["hmjd", "mag", "magerr", "filterid"])
    rows = []

    for i in range(len(table)):
        if len(rows) >= n_samples:
            break

        hmjd = table["hmjd"][i].as_py()
        mag_raw = table["mag"][i].as_py()
        magerr_raw = table["magerr"][i].as_py()
        filterid = table["filterid"][i].as_py()

        mjd = np.asarray(hmjd, dtype=np.float64)
        mag = np.asarray(mag_raw, dtype=np.float32)
        # magerr stored as integers in units of 1e-4 mag
        magerr = np.asarray(magerr_raw, dtype=np.float32) * 1e-4
        band = np.array([FILTERID_TO_BAND.get(f, "g") for f in filterid])

        valid = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
        mjd, mag, magerr, band = mjd[valid], mag[valid], magerr[valid], band[valid]
        if len(mjd) == 0:
            continue

        lc = {"mjd": mjd, "mag": mag, "magerr": magerr, "band": band}
        inp, times, binfo, mask = _preprocess_lc(lc)

        (embedding,) = sess.run(
            ["mean"],
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
    print(f"Saved {len(rows)} test samples to {path}")
