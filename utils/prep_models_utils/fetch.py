import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_fetch(path: str, *, update: bool = False) -> None:
    action = "Updating" if update else "Initializing"
    print(f"{action} submodule {path} ...")
    command = ["git", "submodule", "update", "--init"]
    if update:
        command.append("--remote")
    command.append(path)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    print("Done.")
