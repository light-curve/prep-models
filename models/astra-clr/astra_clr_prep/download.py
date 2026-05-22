"""Download the AstraCLR ONNX file and ZTF sample data from HuggingFace."""

from prep_models_utils.download import download_file

from astra_clr_prep.config import (
    DATA_DIR,
    DATA_FILE,
    HF_DATA_REPO,
    HF_DATA_URL_PATH,
    HF_ONNX_FILENAME,
    HF_REPO,
    WEIGHTS_DIR,
)

_HF_MODEL_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{filename}"
_HF_DATASET_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def run_download(*, force: bool = False) -> None:
    dest = WEIGHTS_DIR / HF_ONNX_FILENAME
    if dest.exists() and not force:
        print(f"ONNX already present at {dest}, skipping.")
    else:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        url = _HF_MODEL_RESOLVE.format(repo=HF_REPO, filename=HF_ONNX_FILENAME)
        print(f"Downloading {HF_ONNX_FILENAME} from {HF_REPO} ...")
        download_file(url, dest)
        print(f"Saved to {dest}")

    if DATA_FILE.exists() and not force:
        print(f"Sample data already present at {DATA_FILE}, skipping.")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = _HF_DATASET_RESOLVE.format(repo=HF_DATA_REPO, path=HF_DATA_URL_PATH)
        print(f"Downloading sample data from {HF_DATA_REPO} ...")
        download_file(url, DATA_FILE)
        print(f"Saved to {DATA_FILE}")
