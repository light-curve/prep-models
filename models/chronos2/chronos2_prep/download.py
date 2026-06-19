"""Cache Chronos 2 weights from HuggingFace."""

from __future__ import annotations

from chronos import Chronos2Pipeline

from chronos2_prep.config import HF_REPO, HF_REVISION


def run_download(*, force: bool = False) -> None:
    print(f"Downloading / verifying {HF_REPO}@{HF_REVISION[:8]} weights ...")
    # from_pretrained caches weights in ~/.cache/huggingface automatically.
    # We call it here so the cache is warm before export.  The pinned revision
    # guarantees identical weights across runs.
    Chronos2Pipeline.from_pretrained(HF_REPO, revision=HF_REVISION, device_map="cpu")
    print("Done.")
