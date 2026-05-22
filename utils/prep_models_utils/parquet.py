from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_OBSERVATION_TYPE = pa.struct(
    [
        pa.field("mjd", pa.float64()),
        pa.field("mag", pa.float32()),
        pa.field("magerr", pa.float32()),
    ]
)


def _build_test_data_table(
    rows: Sequence[dict], embedding_name: str = "embedding_mean"
) -> pa.Table:
    embedding_dim = len(rows[0][embedding_name])

    optional_fields = []
    for name, dtype in [
        ("lcid", pa.string()),
        ("label", pa.int64()),
        ("survey", pa.string()),
    ]:
        if name in rows[0]:
            optional_fields.append(pa.field(name, dtype))

    schema = pa.schema(
        [
            *optional_fields,
            pa.field("lightcurve", pa.list_(_OBSERVATION_TYPE)),
            pa.field(embedding_name, pa.list_(pa.float32(), embedding_dim)),
        ]
    )
    return pa.Table.from_pylist(list(rows), schema=schema)


def save_test_data(
    rows: Sequence[dict], path: Path, embedding_name: str = "embedding_mean"
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_build_test_data_table(rows, embedding_name), path)


def load_test_data(path: Path) -> pa.Table:
    return pq.read_table(path)
