from pathlib import Path

HF_REPO = "amazon/chronos-2"
# Pinned commit of the amazon/chronos-2 HF repo (config.json + model.safetensors)
# so weights are 100% reproducible regardless of upstream changes to `main`.
HF_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
OUTPUT_PREFIX = "chronos2"

_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = _ROOT
WEIGHTS_DIR = _ROOT / "weights"
