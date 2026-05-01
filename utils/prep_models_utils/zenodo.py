"""Zenodo record file listing and targeted download."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

from .download import download_file

# Repo root: prep_models_utils/zenodo.py -> prep_models_utils/ -> utils/ -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_zenodo_files(record_id: str) -> list:
    """Return the file list for a Zenodo record."""
    url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("files", [])


def download_zenodo_file(
    record_id: str,
    key: str,
    dest_dir: Path,
    extract_zip: bool = True,
) -> Path:
    """Download a single file from a Zenodo record by its key.

    If the file is a zip and extract_zip is True, extract into a temp directory
    then move the contents into dest_dir. Returns dest_dir.
    Otherwise download directly to dest_dir and return the file path.
    """
    files = get_zenodo_files(record_id)
    matches = [f for f in files if f.get("key") == key]
    if not matches:
        available = [f.get("key") for f in files]
        raise FileNotFoundError(
            f"Key '{key}' not found in Zenodo record {record_id}.\n"
            f"Available: {available}"
        )

    url = matches[0].get("links", {}).get("self")
    if url is None:
        raise RuntimeError(f"No download URL for '{key}' in record {record_id}.")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if extract_zip and key.endswith(".zip"):
        with tempfile.TemporaryDirectory(dir=_REPO_ROOT, prefix="tmp_") as tmp:
            tmp_zip = Path(tmp) / key
            print(f"Downloading {key} ...")
            download_file(url, tmp_zip)
            print(f"Extracting {key} ...")
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                zf.extractall(tmp)
            tmp_zip.unlink()
            for item in Path(tmp).iterdir():
                target = dest_dir / item.name
                if target.exists():
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                shutil.move(str(item), dest_dir)
        return dest_dir

    dest = dest_dir / key
    print(f"Downloading {key} ...")
    download_file(url, dest)
    return dest
