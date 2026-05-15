from pathlib import Path

from prep_models_utils.astromer import AstromerConfig


# Zenodo record 16410988 contains ANN_clf.* — the full FCATT model
# (encoder retrained on ZTF DR20 g-band + FC classification head).
# We download ANN_clf.* and rename to weights.* so the standard
# astromer1 load_weights path works with expect_partial().
ZENODO_RECORD_ID = "16410988"
ZENODO_DATA_KEY = "ANN_clf.data-00000-of-00001"
ZENODO_INDEX_KEY = "ANN_clf.index"

# Architecture matches the MACHO astromer1 exactly (confirmed from checkpoint).
CONF = {
    "layers": 2,
    "heads": 4,
    "head_dim": 256,
    "dff": 128,
    "base": 1000,
    "dropout": 0.1,
    "use_leak": False,
    "max_obs": 200,
}

MODEL_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = MODEL_DIR / "code"
WEIGHTS_DIR = MODEL_DIR / "weights"

CONFIG = AstromerConfig(
    model_name="Astromer 1 (ZTF DR20 g-band)",
    code_dir=CODE_DIR,
    weights_dir=WEIGHTS_DIR,
    zenodo_record_id=ZENODO_RECORD_ID,
    zenodo_key=ZENODO_DATA_KEY,
    output_prefix="astromer1_ztfdr20",
    test_data_filename="astromer1_ztfdr20_test.parquet",
)
