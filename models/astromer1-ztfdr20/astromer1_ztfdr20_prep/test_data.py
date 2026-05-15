"""Generate test data: ZTF g-band light curves with embeddings from the ZTF DR20 model."""

from pathlib import Path

from prep_models_utils.astromer import run_test_data as run_shared_test_data

from astromer1_ztfdr20_prep.config import CONFIG
from astromer1_ztfdr20_prep.export import _load_model
from astromer1_ztfdr20_prep.preprocess import preprocess_curves as _preprocess_curves
from astromer1_ztfdr20_prep.surveys import ZTF_RECORDS_DIR, load_ztf_curves


def _load_curves(n_samples: int) -> list[dict]:
    return load_ztf_curves(n_samples, ZTF_RECORDS_DIR)


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    run_shared_test_data(
        output_dir,
        n_samples,
        config=CONFIG,
        load_model=_load_model,
        load_curves=_load_curves,
        preprocess_fn=_preprocess_curves,
    )
