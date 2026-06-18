from pathlib import Path

HF_REPO = "amazon/chronos-2"
OUTPUT_PREFIX = "chronos2"

# Default/trace context length. The exported ONNX has a *dynamic* sequence
# axis (any multiple of patch_size 16, up to the model's native 8192), so this
# is only the length used for tracing the export and for generating test data.
# 512 = 32 patches × patch_size 16.
CONTEXT_LENGTH = 512

_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = _ROOT
WEIGHTS_DIR = _ROOT / "weights"
