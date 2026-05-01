"""Download ATCAT pretrained weights and test parquet data."""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import gdown

from atcat_prep.config import ARCHIVE_NAME, DATA_DIR, GDRIVE_FILE_ID, WEIGHTS_DIR

_PREFIXES_TO_EXTRACT = (
    "results/elasticc/CORE/lc_only_cv_0/",
    "data_parquet/split_0/test_",
)


def run_download(*, force: bool = False) -> None:
    checkpoint = (
        WEIGHTS_DIR / "results/elasticc/CORE/lc_only_cv_0/checkpoints/model_40000.pt"
    )
    sample_test = DATA_DIR / "data_parquet/split_0/test_00000.parquet"
    if checkpoint.exists() and sample_test.exists() and not force:
        print(
            "ATCAT checkpoint and test parquet already present. Use --force to re-download."
        )
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / ARCHIVE_NAME
        print("Downloading ATCAT derived data archive from Google Drive ...")
        gdown.download(id=GDRIVE_FILE_ID, output=str(archive_path), quiet=False)

        print("Extracting required ATCAT files ...")
        with tarfile.open(archive_path) as tf:
            members = [
                member
                for member in tf.getmembers()
                if member.name.startswith(_PREFIXES_TO_EXTRACT)
            ]
            if not members:
                raise FileNotFoundError(
                    "Could not find expected ATCAT files in the downloaded archive."
                )
            for member in members:
                if member.name.startswith("results/"):
                    tf.extract(member, path=WEIGHTS_DIR)
                else:
                    tf.extract(member, path=DATA_DIR)

    print(f"Checkpoint ready: {checkpoint}")
    print(f"Test parquet directory ready: {DATA_DIR / 'data_parquet/split_0'}")
