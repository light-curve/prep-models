"""Generate test data: real ZTF BTS supernova light curves + Maven embeddings.

Maven was fine-tuned on ZTF Bright Transient Survey (BTS) light curves.  The
upstream repository ships that sample as `data/ZTFBTS.zip` (the same files as
the `ZTFBTS/` directory of the upstream HuggingFace dataset), so we read it
straight from the submodule and apply the upstream preprocessing
(`src.dataloader.load_lightcurves`).
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import extinction
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from maven_prep.config import (
    BANDS,
    CODE_DIR,
    N_MAX_OBS,
    SEQ_LEN,
    TEST_DATA_FILENAME,
)
from maven_prep.export import MavenEmbedder
from maven_prep.runtime import load_model

ZTFBTS_ZIP = CODE_DIR / "data" / "ZTFBTS.zip"
_TRANSIENT_TABLE = "ZTFBTS/ZTFBTS_TransientTable.csv"
_LIGHT_CURVES_DIR = "ZTFBTS/light-curves/"

# Upstream `load_classes(n_classes=5)`: merged subtypes and the five classes
# used for fine-tuning.
_CLASS_MERGE = {
    "SN Ib": "SN Ibc",
    "SN Ic": "SN Ibc",
    "SN Ib/c": "SN Ibc",
    "SN IIP": "SN II",
}
_CLASSES = ("SN Ia", "SN Ibc", "SN II", "SN IIn", "SLSN-I")

# Upstream `load_lightcurves` effective wavelengths (Angstrom) for the Milky Way
# extinction correction.  The g value is upstream's, reproduced as-is.
_WAVE_EFF = {"g": 1196.25, "R": 6366.38}
_R_V = 3.1

_OBS_TYPE = pa.struct(
    [
        pa.field("mjd", pa.float64()),
        pa.field("mag", pa.float32()),
        pa.field("magerr", pa.float32()),
        pa.field("band", pa.string()),
    ]
)


def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))


def _select_objects(
    zf: zipfile.ZipFile, n_samples: int, rng: np.random.Generator
) -> list[dict]:
    """Pick `n_samples` BTS objects, spread evenly over the five classes.

    Only objects with observations in both bands are eligible, so every
    sample exercises the full two-band layout.
    """
    have_lc = {
        Path(n).stem
        for n in zf.namelist()
        if n.startswith(_LIGHT_CURVES_DIR) and n.endswith(".csv")
    }
    by_class: dict[str, list[dict]] = {c: [] for c in _CLASSES}
    for row in _read_csv(zf, _TRANSIENT_TABLE):
        cls = _CLASS_MERGE.get(row["type"], row["type"])
        if cls in by_class and row["ZTFID"] in have_lc:
            by_class[cls].append({**row, "type": cls})

    selected = []
    for i, cls in enumerate(_CLASSES):
        n_cls = n_samples // len(_CLASSES) + (i < n_samples % len(_CLASSES))
        candidates = sorted(by_class[cls], key=lambda r: r["ZTFID"])
        rng.shuffle(candidates)
        picked = 0
        for row in candidates:
            if picked == n_cls:
                break
            obs = _load_curve(zf, row["ZTFID"])
            if all(any(o["band"] == b for o in obs) for b in BANDS):
                selected.append({"row": row, "obs": obs})
                picked += 1
    return selected


def _load_curve(zf: zipfile.ZipFile, ztfid: str) -> list[dict]:
    rows = _read_csv(zf, f"{_LIGHT_CURVES_DIR}{ztfid}.csv")
    return [
        {
            "mjd": float(r["time"]),
            "mag": float(r["mag"]),
            "magerr": float(r["magerr"]),
            "band": r["band"],
        }
        for r in rows
        if r["band"] in BANDS
    ]


def preprocess(
    obs: list[dict], a_v: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upstream `load_lightcurves` preprocessing for one light curve.

    Per band (R first, then g): correct for Milky Way extinction (CCM89,
    R_V = 3.1), keep at most N_MAX_OBS observations (random subsample without
    replacement if longer), subtract the band's earliest kept observation
    time, zero-pad to N_MAX_OBS and build a validity mask.
    """
    mags, times, masks = [], [], []
    for band in BANDS:
        rows = [o for o in obs if o["band"] == band]
        t = np.array([o["mjd"] for o in rows], dtype=np.float64)
        m = np.array([o["mag"] for o in rows], dtype=np.float64)
        m -= extinction.ccm89(np.array([_WAVE_EFF[band]]), a_v, _R_V)[0]
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
    samples: list[dict],
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    mean: np.ndarray,
    clip: np.ndarray,
    path: Path,
) -> None:
    mag, time, mask = inputs
    schema = pa.schema(
        [
            pa.field("object_id", pa.string()),
            pa.field("class", pa.string()),
            pa.field("a_v", pa.float32()),
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
            "object_id": sample["row"]["ZTFID"],
            "class": sample["row"]["type"],
            "a_v": float(sample["row"]["A_V"]),
            "lightcurve": sample["obs"],
            "input_mag": mag[i].tolist(),
            "input_time": time[i].tolist(),
            "input_mask": mask[i].tolist(),
            "embedding_mean": mean[i].tolist(),
            "embedding_clip": clip[i].tolist(),
        }
        for i, sample in enumerate(samples)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def run_test_data(output_dir: Path, n_samples: int = 10, *, seed: int = 42) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if not ZTFBTS_ZIP.exists():
        raise FileNotFoundError(
            f"{ZTFBTS_ZIP} not found. Run 'prep-models maven fetch' first."
        )
    print(f"Selecting {n_samples} ZTF BTS supernovae from {ZTFBTS_ZIP} ...")
    with zipfile.ZipFile(ZTFBTS_ZIP) as zf:
        samples = _select_objects(zf, n_samples, rng)
    prepped = [preprocess(s["obs"], float(s["row"]["A_V"]), rng) for s in samples]
    inputs = tuple(np.stack(arrays) for arrays in zip(*prepped))

    embedder = MavenEmbedder(load_model())
    embedder.eval()
    with torch.no_grad():
        mean, clip, _ = embedder(*(torch.from_numpy(a) for a in inputs))

    out_path = output_dir / TEST_DATA_FILENAME
    _save(samples, inputs, mean.numpy(), clip.numpy(), out_path)
    print(f"Saved {len(samples)} samples → {out_path}")
