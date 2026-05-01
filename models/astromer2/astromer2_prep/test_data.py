"""Generate test data: real light curves with embeddings from original model.

Mix of Alcock/MACHO (pretraining survey) and ATLAS (cross-survey).
"""

from pathlib import Path

from prep_models_utils.astromer import run_test_data as run_shared_test_data
from prep_models_utils.astromer.surveys import load_mixed_curves

from astromer2_prep.config import CONFIG
from astromer2_prep.export import _load_model
from astromer2_prep.preprocess import preprocess_curves as _preprocess_curves

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ALCOCK_RECORDS_DIR = _DATA_DIR / "alcock-records"
ATLAS_RECORDS_DIR = _DATA_DIR / "atlas-records"


def _load_curves(n_samples: int) -> list[dict]:
    return load_mixed_curves(n_samples, ALCOCK_RECORDS_DIR, ATLAS_RECORDS_DIR)


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    run_shared_test_data(
        output_dir,
        n_samples,
        config=CONFIG,
        load_model=_load_model,
        load_curves=_load_curves,
        preprocess_fn=_preprocess_curves,
    )
