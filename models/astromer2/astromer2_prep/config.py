from pathlib import Path

from prep_models_utils.astromer import AstromerConfig


ZENODO_RECORD_ID = "18207945"
ZENODO_KEY = "pt_macho_v2_2025.zip"

MODEL_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = MODEL_DIR / "code"
WEIGHTS_DIR = MODEL_DIR / "weights"

CONFIG = AstromerConfig(
    model_name="Astromer 2",
    code_dir=CODE_DIR,
    weights_dir=WEIGHTS_DIR,
    zenodo_record_id=ZENODO_RECORD_ID,
    zenodo_key=ZENODO_KEY,
    output_prefix="astromer2",
    test_data_filename="astromer2_test.parquet",
)
