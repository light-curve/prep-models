from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = MODEL_DIR / "code"
WEIGHTS_DIR = MODEL_DIR / "weights"
DATA_DIR = MODEL_DIR / "data"

# Google Drive file ID for results_paper.zip from the official ATAT repository.
# Source: models/atat/code/get_important_files.sh
GDRIVE_FILE_ID = "1teIi3GfPbYOZXaIHRAa_OCPTY9QYXTU1"

# After extraction, checkpoints for the LC-only model live here (one per seed 0-4).
# We pick seed 0 by default; each seed gives an independent trained model.
CHECKPOINT_GLOB = "results_paper/lc/Exp_cfg_-arch=lc-seed=0/my_best_checkpoint*.ckpt"

# ELASTICC FITS data (one HEAD+PHOT pair used for test data generation)
ELASTICC_BASE_URL = (
    "https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/TRAINING_SAMPLES"
)
ELASTICC_CLASS_DIR = "ELASTICC_TRAIN_SNIa-SALT2"
ELASTICC_FILE_STEM = "ELASTICC_TRAIN_NONIaMODEL0-0001"

# ELASTICC dataset constants (from datasets.py in the upstream code).
DATASET_CHANNEL = 6  # photometric bands: u, g, r, i, z, Y
SEQ_LEN = 65  # max observations per band
T_MAX = 1500  # time normalization constant for the time modulator
EMBED_DIM = 192  # head_dim=48 × num_heads=4
