from pathlib import Path

HF_REPO = "amazon/chronos-2"
OUTPUT_PREFIX = "chronos2"

# Fixed context length for the exported ONNX model.
# 512 = 32 patches × patch_size 16.  Covers most light curves;
# longer series are truncated to the last CONTEXT_LENGTH observations.
CONTEXT_LENGTH = 512

_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = _ROOT
WEIGHTS_DIR = _ROOT / "weights"
