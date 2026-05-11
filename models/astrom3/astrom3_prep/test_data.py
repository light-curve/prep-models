"""Generate AstroM3 test data from the upstream HuggingFace dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from astrom3_prep.config import (
    CLASS_NAMES,
    HF_DATA_REPO,
    SEQ_LEN,
    TEST_DATA_FILENAME,
)
from astrom3_prep.export import _AstroM3Embedder, load_export_model


def _load_rows(n_samples: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATA_REPO, name="full_0", split="test")
    rows = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        photometry = item["photometry"]
        if not isinstance(photometry, torch.Tensor):
            photometry = torch.tensor(photometry, dtype=torch.float32)
        else:
            photometry = photometry.float()

        # Pad or trim to SEQ_LEN
        n = photometry.shape[0]
        if n >= SEQ_LEN:
            photo_padded = photometry[:SEQ_LEN]
            mask = torch.ones(SEQ_LEN, dtype=torch.float32)
        else:
            pad = torch.zeros(SEQ_LEN - n, photometry.shape[1], dtype=torch.float32)
            photo_padded = torch.cat([photometry, pad], dim=0)
            mask = torch.cat(
                [
                    torch.ones(n, dtype=torch.float32),
                    torch.zeros(SEQ_LEN - n, dtype=torch.float32),
                ]
            )

        label = int(item["label"]) if "label" in item else 0
        rows.append(
            {
                "photometry": photo_padded,
                "mask": mask,
                "label": label,
                "object_id": str(item.get("object_id", item.get("id", i))),
            }
        )
    return rows


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {n_samples} samples from {HF_DATA_REPO} ...")
    rows = _load_rows(n_samples)

    x_enc = torch.stack([r["photometry"] for r in rows])  # (N, 200, 9)
    mask = torch.stack([r["mask"] for r in rows])  # (N, 200)

    model = load_export_model()
    embedder = _AstroM3Embedder(model)
    embedder.eval()
    with torch.no_grad():
        mean_emb, _, _ = embedder(x_enc, mask)
    embeddings = mean_emb.numpy()  # (N, 128)

    out_path = output_dir / TEST_DATA_FILENAME
    _save(rows, embeddings, out_path)
    print(f"Saved {len(rows)} samples to {out_path}")


def _save(rows: list[dict], embeddings: np.ndarray, path: Path) -> None:
    embed_dim = embeddings.shape[1]
    schema = pa.schema(
        [
            pa.field("object_id", pa.string()),
            pa.field("label", pa.int32()),
            pa.field("class_name", pa.string()),
            pa.field("photometry", pa.list_(pa.list_(pa.float32(), SEQ_LEN))),
            pa.field("mask", pa.list_(pa.float32(), SEQ_LEN)),
            pa.field("embedding_mean", pa.list_(pa.float32(), embed_dim)),
        ]
    )

    table_rows = []
    for i, row in enumerate(rows):
        label = row["label"]
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
        photo_np = row["photometry"].numpy()  # (200, 9)
        table_rows.append(
            {
                "object_id": row["object_id"],
                "label": label,
                "class_name": class_name,
                # Store as list of per-feature arrays for easy inspection
                "photometry": [
                    photo_np[:, ch].tolist() for ch in range(photo_np.shape[1])
                ],
                "mask": row["mask"].numpy().tolist(),
                "embedding_mean": embeddings[i].tolist(),
            }
        )

    pq.write_table(pa.Table.from_pylist(table_rows, schema=schema), path)
