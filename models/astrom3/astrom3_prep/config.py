from __future__ import annotations

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]

# HuggingFace source repositories
HF_MODEL_REPO = "AstroMLCore/AstroM3-CLIP-photo"
HF_DATA_REPO = "AstroMLCore/AstroM3Processed"

# Informer architecture (from config.json on HuggingFace)
SEQ_LEN = 200
ENC_IN = 9
D_MODEL = 128
D_FF = 512
N_HEADS = 4
E_LAYERS = 8
FACTOR = 1
DROPOUT = 0.14904317667318695
ACTIVATION = "gelu"

OUTPUT_PREFIX = "astrom3"
TEST_DATA_FILENAME = "astrom3_test.parquet"
