from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = MODEL_DIR / "weights"
DATA_DIR = MODEL_DIR / "data"

HF_REPO = "ashrot/astra-clr-base"
HF_ONNX_FILENAME = "astra-clr.onnx"
OUTPUT_FILENAME = "astra_clr.onnx"

HF_DATA_REPO = "snad-space/astra-zubercaldr16_gaiadr3vclassre"
# URL path uses %3D-encoded '=' signs required by the HF dataset server
HF_DATA_URL_PATH = "hats/dataset/Norder%3D0/Dir%3D0/Npix%3D8.parquet"
DATA_FILE = DATA_DIR / "Npix=8.parquet"
