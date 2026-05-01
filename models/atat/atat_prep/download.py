"""Download ATAT pretrained weights and ELASTICC sample data."""

import glob
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import gdown

from atat_prep.config import (
    CHECKPOINT_GLOB,
    DATA_DIR,
    ELASTICC_BASE_URL,
    ELASTICC_CLASS_DIR,
    ELASTICC_FILE_STEM,
    GDRIVE_FILE_ID,
    WEIGHTS_DIR,
)


def run_download(*, force: bool = False) -> None:
    _download_elasticc_fits(force=force)

    dest = WEIGHTS_DIR / "checkpoint.ckpt"
    if dest.exists() and not force:
        print(f"Checkpoint already present at {dest}. Use --force to re-download.")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "results_paper.zip"

        print("Downloading results_paper.zip from Google Drive ...")
        gdown.download(id=GDRIVE_FILE_ID, output=str(zip_path), quiet=False)

        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        matches = sorted(glob.glob(str(Path(tmp) / CHECKPOINT_GLOB)))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint found matching {CHECKPOINT_GLOB} inside the archive. "
                "The zip may have a different layout than expected."
            )

        shutil.copy2(matches[0], dest)

    print(f"Checkpoint saved to {dest}")


def _download_elasticc_fits(*, force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("HEAD.FITS.gz", "PHOT.FITS.gz"):
        dest = DATA_DIR / f"{ELASTICC_FILE_STEM}_{suffix}"
        if dest.exists() and not force:
            print(f"{dest.name} already present, skipping.")
            continue
        url = f"{ELASTICC_BASE_URL}/{ELASTICC_CLASS_DIR}/{ELASTICC_FILE_STEM}_{suffix}"
        print(f"Downloading {dest.name} ...")
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    print("ELASTICC FITS pair ready.")
