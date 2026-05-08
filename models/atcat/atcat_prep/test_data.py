"""Generate ATCAT test parquet from upstream derived Parquet data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from atcat_prep.config import TEST_DATA_FILENAME
from atcat_prep.export import _ATCATEmbedder, load_export_model
from atcat_prep.runtime import add_code_to_path, test_parquet_dir


def _load_rows(n_samples: int) -> list[dict]:
    add_code_to_path()
    from atcat.elasticc import elasticc_schema, lsst_colors

    parquet_dir = test_parquet_dir()
    files = sorted(parquet_dir.glob("test_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No ATCAT test parquet files found in {parquet_dir}. Run 'prep-models atcat download' first."
        )

    import polars as pl

    df = pl.read_parquet(files)
    rows = df.head(n_samples).to_dicts()
    for row in rows:
        row["class_name"] = elasticc_schema.ATAT_FINE_CLASSES_WITHOUT_UNKNOWN[
            row["label_atat_fine_class"]
        ]
        row["band_names"] = lsst_colors.colors
    return rows


def _pad_array(values: list, *, seq_len: int, dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if len(arr) > seq_len:
        raise ValueError(
            f"Sequence length {len(arr)} exceeds fixed ATCAT length {seq_len}"
        )
    out = np.zeros(seq_len, dtype=dtype)
    out[: len(arr)] = arr
    return out


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(n_samples)
    seq_len = 243

    flux = np.stack(
        [_pad_array(row["flux"], seq_len=seq_len, dtype=np.float32) for row in rows]
    )
    flux_err = np.stack(
        [_pad_array(row["flux_err"], seq_len=seq_len, dtype=np.float32) for row in rows]
    )
    time = np.stack(
        [_pad_array(row["time"], seq_len=seq_len, dtype=np.float32) for row in rows]
    )
    channel_index = np.stack(
        [
            _pad_array(row["channel_index"], seq_len=seq_len, dtype=np.int64)
            for row in rows
        ]
    )
    mask = np.stack(
        [
            np.concatenate(
                [
                    np.ones(len(row["flux"]), dtype=bool),
                    np.zeros(seq_len - len(row["flux"]), dtype=bool),
                ]
            )
            for row in rows
        ]
    )

    model = load_export_model()
    embedder = _ATCATEmbedder(model)
    embedder.eval()
    with torch.no_grad():
        embedding_last, _mean, _sequence = embedder(
            torch.from_numpy(flux),
            torch.from_numpy(flux_err),
            torch.from_numpy(time),
            torch.from_numpy(mask),
            torch.from_numpy(channel_index),
        )
        embeddings = embedding_last.numpy()

    out_path = output_dir / TEST_DATA_FILENAME
    _save(rows, embeddings, out_path)
    print(f"Saved {len(rows)} samples to {out_path}")


def _save(rows: list[dict], embeddings: np.ndarray, path: Path) -> None:
    embed_dim = embeddings.shape[1]
    lc_type = pa.struct(
        [
            pa.field("band", pa.string()),
            pa.field("time", pa.list_(pa.float32())),
            pa.field("flux", pa.list_(pa.float32())),
            pa.field("flux_err", pa.list_(pa.float32())),
        ]
    )
    schema = pa.schema(
        [
            pa.field("lcid", pa.string()),
            pa.field("label_atat_fine_class", pa.int32()),
            pa.field("class_name", pa.string()),
            pa.field("lightcurve", pa.list_(lc_type)),
            pa.field("embedding_last", pa.list_(pa.float32(), embed_dim)),
        ]
    )

    table_rows = []
    for i, row in enumerate(rows):
        by_band = []
        for band_idx, band_name in enumerate(row["band_names"]):
            keep = [
                j for j, value in enumerate(row["channel_index"]) if value == band_idx
            ]
            if not keep:
                continue
            by_band.append(
                {
                    "band": band_name,
                    "time": [float(row["time"][j]) for j in keep],
                    "flux": [float(row["flux"][j]) for j in keep],
                    "flux_err": [float(row["flux_err"][j]) for j in keep],
                }
            )
        table_rows.append(
            {
                "lcid": str(row["SNID"]),
                "label_atat_fine_class": int(row["label_atat_fine_class"]),
                "class_name": row["class_name"],
                "lightcurve": by_band,
                "embedding_last": embeddings[i].tolist(),
            }
        )

    pq.write_table(pa.Table.from_pylist(table_rows, schema=schema), path)
