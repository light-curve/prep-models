"""Download the AstraCLR ONNX file and ZTF sample data from HuggingFace."""

import shutil

from huggingface_hub import hf_hub_download

from astra_clr_prep.config import (
    DATA_DIR,
    DATA_FILE,
    HF_DATA_FILENAME,
    HF_DATA_REPO,
    HF_ONNX_FILENAME,
    HF_REPO,
    WEIGHTS_DIR,
)


def run_download(*, force: bool = False) -> None:
    dest = WEIGHTS_DIR / HF_ONNX_FILENAME
    if dest.exists() and not force:
        print(f"ONNX already present at {dest}, skipping.")
    else:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {HF_ONNX_FILENAME} from {HF_REPO} ...")
        cached = hf_hub_download(repo_id=HF_REPO, filename=HF_ONNX_FILENAME)
        shutil.copy2(cached, dest)
        print(f"Saved to {dest}")

    if DATA_FILE.exists() and not force:
        print(f"Sample data already present at {DATA_FILE}, skipping.")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading sample data from {HF_DATA_REPO} ...")
        cached = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=HF_DATA_FILENAME,
            repo_type="dataset",
        )
        shutil.copy2(cached, DATA_FILE)
        print(f"Saved to {DATA_FILE}")
