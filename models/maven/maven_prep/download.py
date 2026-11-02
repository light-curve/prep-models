"""Download the Maven fine-tuned CLIP checkpoint.

The checkpoints are committed to the upstream repository, so they are fetched
from raw.githubusercontent.com at the commit the submodule is pinned to.
"""

from __future__ import annotations

from prep_models_utils import download_file

from maven_prep.config import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    RUN_DIR,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    WEIGHTS_DIR,
)


def run_download(*, force: bool = False) -> None:
    dest = WEIGHTS_DIR / CHECKPOINT_NAME
    if dest.exists() and not force:
        print(f"Weights already present at {dest} (pass --force to re-download).")
        return
    dest.unlink(missing_ok=True)

    url = (
        f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/"
        f"{RUN_DIR}/{CHECKPOINT_NAME}"
    )
    print(f"Downloading {url} ...")
    download_file(url, dest, sha256=CHECKPOINT_SHA256)
    print(f"Saved to {dest}")
