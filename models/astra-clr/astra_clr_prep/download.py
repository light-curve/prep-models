"""Download the AstraCLR ONNX file from HuggingFace."""

from prep_models_utils.download import download_file

from astra_clr_prep.config import HF_ONNX_FILENAME, HF_REPO, WEIGHTS_DIR

_HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{filename}"


def run_download(*, force: bool = False) -> None:
    dest = WEIGHTS_DIR / HF_ONNX_FILENAME
    if dest.exists() and not force:
        print(f"ONNX already present at {dest}, skipping.")
        return
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    url = _HF_RESOLVE.format(repo=HF_REPO, filename=HF_ONNX_FILENAME)
    print(f"Downloading {HF_ONNX_FILENAME} from {HF_REPO} ...")
    download_file(url, dest)
    print(f"Saved to {dest}")
