"""Generate test data: real ELASTICC light curves with ATAT embeddings.

Uses the ATAT upstream FITS reader and preprocessing pipeline directly:
- GetElasticcData.processing_fits  (get_fits_elasticc_data.py)
- separate_by_filter / normalizing_time / create_mask  (get_lc_md.py)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from atat_prep.config import CODE_DIR, DATA_DIR, ELASTICC_FILE_STEM
from atat_prep.export import _ATATEmbedder, load_encoder

TEST_DATA_FILENAME = "atat_test.parquet"

_BANDS = ["u", "g", "r", "i", "z", "Y"]


def _add_code_to_path() -> None:
    code = str(CODE_DIR)
    if code not in sys.path:
        sys.path.insert(0, code)


def _load_curves(n_samples: int) -> list[dict]:
    """Read FITS pair and preprocess using ATAT's own pipeline functions."""
    _add_code_to_path()
    from src.data.get_fits_elasticc_data import GetElasticcData
    from src.data.get_lc_md import create_mask, normalizing_time, separate_by_filter

    head_path = DATA_DIR / f"{ELASTICC_FILE_STEM}_HEAD.FITS.gz"
    phot_path = DATA_DIR / f"{ELASTICC_FILE_STEM}_PHOT.FITS.gz"

    if not head_path.exists() or not phot_path.exists():
        raise FileNotFoundError(
            f"ELASTICC FITS files not found in {DATA_DIR}. "
            "Run 'prep-models atat download' first."
        )

    # processing_fits uses relative paths prefixed with './'; run from DATA_DIR
    reader = GetElasticcData.__new__(GetElasticcData)
    prev_dir = os.getcwd()
    os.chdir(DATA_DIR)
    try:
        df_lc, df_header = reader.processing_fits((head_path.name, phot_path.name))
    finally:
        os.chdir(prev_dir)

    curves = []
    for snid, group in list(df_lc.groupby("SNID"))[:n_samples]:
        header_row = df_header[df_header["SNID"] == snid].iloc[0]

        mjd = group["MJD"].tolist()
        flux = group["FLUXCAL"].tolist()
        bands = group["BAND"].tolist()

        time_mat = separate_by_filter(mjd, bands).astype(np.float32)
        data_mat = separate_by_filter(flux, bands).astype(np.float32)
        time_mat = normalizing_time(time_mat).astype(np.float32)
        mask_mat = create_mask(data_mat).astype(np.float32)

        curves.append(
            {
                "lcid": str(snid),
                "sntype": int(header_row["SNTYPE"]),
                "data": data_mat,
                "time": time_mat,
                "mask": mask_mat,
            }
        )

    return curves


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {n_samples} ELASTICC light curves ...")
    curves = _load_curves(n_samples)

    data_t = torch.from_numpy(np.stack([c["data"] for c in curves]))
    time_t = torch.from_numpy(np.stack([c["time"] for c in curves]))
    mask_t = torch.from_numpy(np.stack([c["mask"] for c in curves]))

    print("Loading model and computing token embeddings ...")
    encoder = load_encoder()
    embedder = _ATATEmbedder(encoder)
    embedder.eval()

    with torch.no_grad():
        embedding_token, _, _ = embedder(data_t, time_t, mask_t)
        embeddings = embedding_token.numpy()

    out_path = output_dir / TEST_DATA_FILENAME
    _save(curves, embeddings, out_path)
    print(f"Saved {len(curves)} samples to {out_path}")


def _save(curves: list[dict], embeddings: np.ndarray, path: Path) -> None:
    embed_dim = embeddings.shape[1]
    lc_type = pa.struct(
        [
            pa.field("band", pa.string()),
            pa.field("time", pa.list_(pa.float32())),
            pa.field("flux", pa.list_(pa.float32())),
        ]
    )
    schema = pa.schema(
        [
            pa.field("lcid", pa.string()),
            pa.field("sntype", pa.int32()),
            pa.field("lightcurve", pa.list_(lc_type)),
            pa.field("embedding_token", pa.list_(pa.float32(), embed_dim)),
        ]
    )
    rows = []
    for i, c in enumerate(curves):
        lc = []
        for b, band in enumerate(_BANDS):
            mask = c["mask"][:, b].astype(bool)
            if mask.any():
                lc.append(
                    {
                        "band": band,
                        "time": c["time"][:, b][mask].tolist(),
                        "flux": c["data"][:, b][mask].tolist(),
                    }
                )
        rows.append(
            {
                "lcid": c["lcid"],
                "sntype": c["sntype"],
                "lightcurve": lc,
                "embedding_token": embeddings[i].tolist(),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
