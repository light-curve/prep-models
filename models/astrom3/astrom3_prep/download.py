"""Download AstroM3 CLIP-photo pretrained weights from HuggingFace."""

from __future__ import annotations

import shutil

from astrom3_prep.config import HF_MODEL_REPO
from astrom3_prep.runtime import _WEIGHTS_DIR


def run_download(*, force: bool = False) -> None:
    from huggingface_hub import hf_hub_download

    dest = _WEIGHTS_DIR / "model.safetensors"
    if dest.exists() and not force:
        print(f"Weights already present at {dest} (pass --force to re-download).")
        return

    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights from {HF_MODEL_REPO} ...")
    cached = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.safetensors")
    shutil.copy(cached, dest)
    print(f"Saved to {dest}")
