from pathlib import Path

HF_REPO = "amazon/chronos-2"
OUTPUT_PREFIX = "chronos2"

_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = _ROOT
WEIGHTS_DIR = _ROOT / "weights"
