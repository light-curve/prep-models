"""Cache Chronos 2 weights from HuggingFace."""

from __future__ import annotations

from chronos import Chronos2Pipeline

from chronos2_prep.config import HF_REPO


def run_download(*, force: bool = False) -> None:
    print(f"Downloading / verifying {HF_REPO} weights from HuggingFace ...")
    # from_pretrained caches weights in ~/.cache/huggingface automatically.
    # We call it here so the cache is warm before export.
    Chronos2Pipeline.from_pretrained(HF_REPO, device_map="cpu")
    print("Done.")
