"""Cache Chronos-Bolt weights from HuggingFace."""

from __future__ import annotations

from chronos import ChronosBoltPipeline

from chronos_bolt_prep.config import hf_repo, hf_revision


def run_download(*, force: bool = False, size: str) -> None:
    repo, rev = hf_repo(size), hf_revision(size)
    print(f"Downloading / verifying {repo}@{rev[:8]} weights ...")
    # from_pretrained caches weights in ~/.cache/huggingface automatically.
    # The pinned revision guarantees identical weights across runs.
    ChronosBoltPipeline.from_pretrained(repo, revision=rev, device_map="cpu")
    print("Done.")
