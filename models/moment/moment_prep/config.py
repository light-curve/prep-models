from pathlib import Path

# One ONNX model per MOMENT-1 size.  All three share the same architecture,
# interface, and preprocessing — only the weights (and `d_model`) differ.
#
# Each entry pins the exact HuggingFace commit so the weights are 100%
# reproducible regardless of upstream changes to `main`.
SIZES: dict[str, dict[str, str]] = {
    "small": {
        "hf_repo": "AutonLab/MOMENT-1-small",
        "hf_revision": "411e288267f82cce86296dbe4d6c8bc533cc162f",
    },
    "base": {
        "hf_repo": "AutonLab/MOMENT-1-base",
        "hf_revision": "5e44b0ea26376a176360f87831124e018f876d96",
    },
    "large": {
        "hf_repo": "AutonLab/MOMENT-1-large",
        "hf_revision": "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc",
    },
}

DEFAULT_SIZE = "base"

# MOMENT uses a fixed context window of 512 observations split into 64
# non-overlapping patches of 8.  Unlike the Chronos family, the sequence
# length is *not* a dynamic axis.
SEQ_LEN = 512
PATCH_LEN = 8

_ROOT = Path(__file__).resolve().parents[1]  # models/moment
MODEL_DIR = _ROOT


def hf_repo(size: str) -> str:
    return SIZES[size]["hf_repo"]


def hf_revision(size: str) -> str:
    return SIZES[size]["hf_revision"]


def output_prefix(size: str) -> str:
    return f"moment1-{size}"


def out_dir(size: str) -> Path:
    """Per-size output root: models/moment/out/<size>/."""
    return _ROOT / "out" / size
