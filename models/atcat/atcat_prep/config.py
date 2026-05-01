from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = MODEL_DIR / "code"
WEIGHTS_DIR = MODEL_DIR / "weights"
DATA_DIR = MODEL_DIR / "data"

GDRIVE_FILE_ID = "1xdVVnvDf7hhqhvbx-ZbywaOT1Sz9REKf"
ARCHIVE_NAME = "atcat_derived_data.tar"

RESULTS_SUBDIR = Path("results/elasticc/CORE/lc_only_cv_0")
CHECKPOINT_PATH = RESULTS_SUBDIR / "checkpoints/model_40000.pt"
CONFIG_JSON_PATH = RESULTS_SUBDIR / "config.json"

TEST_PARQUET_GLOB = "data_parquet/split_0/test_*.parquet"

OUTPUT_PREFIX = "atcat"
TEST_DATA_FILENAME = "atcat_test.parquet"
