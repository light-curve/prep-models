from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = MODEL_DIR / "weights"

HF_REPO = "ashrot/astra-clr-base"
HF_ONNX_FILENAME = "astra-clr.onnx"
OUTPUT_FILENAME = "astra_clr.onnx"
