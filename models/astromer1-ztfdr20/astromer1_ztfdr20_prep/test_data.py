"""Generate test data: ZTF g-band light curves with embeddings from the ZTF DR20 model."""

from pathlib import Path

import numpy as np
import pandas as pd
from prep_models_utils.astromer import run_test_data as run_shared_test_data

from astromer1_ztfdr20_prep.config import CONFIG, ZTF_RECORDS_DIR, ZTF_RECORDS_FILENAME
from astromer1_ztfdr20_prep.export import _load_model
from astromer1_ztfdr20_prep.preprocess import preprocess_curves as _preprocess_curves


def _load_curves(n_samples: int) -> list[dict]:
    """Load cached ZTF DR17 g-band light curves (written by the download command)."""
    path = ZTF_RECORDS_DIR / ZTF_RECORDS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'prep-models astromer1-ztfdr20 download' first."
        )

    df = pd.read_parquet(path)
    curves = []
    for oid, grp in df.groupby("oid"):
        grp = grp.sort_values("mjd")
        curves.append(
            {
                "lcid": str(oid),
                "survey": "ztf_dr17_g",
                "mjd": grp["mjd"].to_numpy(dtype=np.float32),
                "magnitude": grp["mag"].to_numpy(dtype=np.float32),
                "error": grp["magerr"].to_numpy(dtype=np.float32),
            }
        )
        if len(curves) >= n_samples:
            break

    print(f"Loaded {len(curves)} ZTF DR17 g-band light curves")
    return curves


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    run_shared_test_data(
        output_dir,
        n_samples,
        config=CONFIG,
        load_model=_load_model,
        load_curves=_load_curves,
        preprocess_fn=_preprocess_curves,
    )
