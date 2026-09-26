from __future__ import annotations

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = MODEL_DIR / "code"
WEIGHTS_DIR = MODEL_DIR / "weights"

# Upstream repository and the commit the submodule is pinned to.  The
# checkpoints are committed to the upstream repo itself (no LFS), so the raw
# URL at this commit is a stable download location.
UPSTREAM_REPO = "ThomasHelfer/multimodal-supernovae"
UPSTREAM_COMMIT = "1f571aa9311eccce6358bfa1ff0fa5e8212d1422"

# Maven = CLIP pre-training on simulated (noiseless) ZTF light curves + spectra,
# then CLIP fine-tuning on ZTF BTS observations.  The upstream sweep trains one
# run per stratified k-fold (5 folds); we export fold 0.  Following upstream
# `src.utils.get_checkpoint_paths` (used by `evaluate_models.py`), the
# checkpoint with the smallest epoch number in the run directory is the one
# evaluated in the paper.
RUN_DIR = "models/clip_noiselesssimpretrain_clipreal/gallant-sweep-1"
CHECKPOINT_NAME = "epoch=25-step=3042.ckpt"
CHECKPOINT_SHA256 = "c4aac38a2be0623e1110de7e182b7328be791cdc8c07443d663de78e27e060db"

# Light-curve encoder hyperparameters (from RUN_DIR/config.yaml).
EMB = 64
HEADS = 8
TRANSFORMER_DEPTH = 5
N_OUT = 32
TIME_NORM = 20583.369161312577
AGG = "mean"
# LightCurveImageCLIP default (not overridden by the training scripts).
ENC_DIM = 128

# ZTF bands, in the order they are concatenated along the sequence axis
# (upstream `load_lightcurves`: bands = ["R", "g"]).
BANDS = ("R", "g")
NBAND = len(BANDS)
# Upstream `load_data(max_data_len_lc=100)` default: observations per band.
N_MAX_OBS = 100
SEQ_LEN = NBAND * N_MAX_OBS

OUTPUT_PREFIX = "maven"
TEST_DATA_FILENAME = "maven_test.parquet"
