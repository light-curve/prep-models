import hashlib
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


def download_file(
    url: str,
    dest: Path,
    sha256: Optional[str] = None,
    chunk_size: int = 1 << 20,
) -> Path:
    """Stream-download url to dest, with optional SHA-256 verification.

    Skips download if dest already exists and hash matches (or no hash given).
    Returns dest path.
    """
    dest = Path(dest)
    if dest.exists() and sha256 is None:
        return dest
    if dest.exists() and sha256 is not None and _sha256(dest) == sha256:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0)) or None

    with (
        tmp.open("wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    if sha256 is not None:
        got = _sha256(tmp)
        if got != sha256:
            tmp.unlink()
            raise ValueError(
                f"SHA-256 mismatch for {dest.name}: expected {sha256}, got {got}"
            )

    tmp.rename(dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
