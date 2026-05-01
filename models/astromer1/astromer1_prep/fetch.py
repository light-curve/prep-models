import subprocess
from pathlib import Path

SUBMODULE_PATH = "models/astromer1/code"
# Astromer 1 code lives on the v1.0 tag of the shared upstream repo.
# Astromer 2 uses a later commit on the same repo; pinning here prevents
# fetch from accidentally advancing to the v2 codebase.
PINNED_TAG = "v1.0"

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_fetch(*, update: bool = False) -> None:
    from prep_models_utils import run_fetch as run_shared_fetch

    run_shared_fetch(SUBMODULE_PATH, update=update)

    # Always pin to the correct tag regardless of what the submodule index says
    code_dir = REPO_ROOT / SUBMODULE_PATH
    current = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=code_dir,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != PINNED_TAG:
        print(f"Pinning models/astromer1/code to {PINNED_TAG} ...")
        subprocess.run(["git", "checkout", PINNED_TAG], cwd=code_dir, check=True)
        print(f"Checked out {PINNED_TAG}.")
