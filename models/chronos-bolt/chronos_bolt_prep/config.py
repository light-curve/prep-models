from pathlib import Path

# One ONNX model per Chronos-Bolt size.  All four share the same architecture,
# interface, and preprocessing — only the weights (and `d_model`) differ.
#
# Each entry pins the exact HuggingFace commit so the weights are 100%
# reproducible regardless of upstream changes to `main`.
SIZES: dict[str, dict[str, str]] = {
    "tiny": {
        "hf_repo": "amazon/chronos-bolt-tiny",
        "hf_revision": "a0e552de83495b5c28c14c71c374f3e33280b340",
    },
    "mini": {
        "hf_repo": "amazon/chronos-bolt-mini",
        "hf_revision": "251268337516a88e253628c43e1d26ec577b376b",
    },
    "small": {
        "hf_repo": "amazon/chronos-bolt-small",
        "hf_revision": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
    },
    "base": {
        "hf_repo": "amazon/chronos-bolt-base",
        "hf_revision": "5d9f166d69f47aef3401367a7b842e78fe97b121",
    },
}

DEFAULT_SIZE = "base"

_ROOT = Path(__file__).resolve().parents[1]  # models/chronos-bolt
MODEL_DIR = _ROOT


def hf_repo(size: str) -> str:
    return SIZES[size]["hf_repo"]


def hf_revision(size: str) -> str:
    return SIZES[size]["hf_revision"]


def output_prefix(size: str) -> str:
    return f"chronos-bolt-{size}"


def out_dir(size: str) -> Path:
    """Per-size output root: models/chronos-bolt/out/<size>/."""
    return _ROOT / "out" / size
