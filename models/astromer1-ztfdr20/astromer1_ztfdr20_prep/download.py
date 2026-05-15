"""Download Astromer 1 ZTF DR20 g-band weights and ZTF DR17 survey records."""

import json

from prep_models_utils.zenodo import download_zenodo_file

from astromer1_ztfdr20_prep.config import (
    CONF,
    WEIGHTS_DIR,
    ZENODO_DATA_KEY,
    ZENODO_INDEX_KEY,
    ZENODO_RECORD_ID,
)
from astromer1_ztfdr20_prep.surveys import ZTF_RECORDS_DIR, download_ztf_records


def _download_weights(*, force: bool = False) -> None:
    """Download ANN_clf checkpoint from Zenodo and rename to weights.*."""
    weights_index = WEIGHTS_DIR / "weights.index"
    if weights_index.exists() and not force:
        print(f"Weights already present at {WEIGHTS_DIR}, skipping.")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for zenodo_key, local_name in [
        (ZENODO_DATA_KEY, "weights.data-00000-of-00001"),
        (ZENODO_INDEX_KEY, "weights.index"),
    ]:
        dest = download_zenodo_file(
            ZENODO_RECORD_ID, zenodo_key, WEIGHTS_DIR, extract_zip=False
        )
        dest.rename(WEIGHTS_DIR / local_name)

    conf_path = WEIGHTS_DIR / "conf.json"
    with open(conf_path, "w") as f:
        json.dump(CONF, f, indent=4)
    print(f"Weights saved to {WEIGHTS_DIR}")


def run_download(*, force: bool = False) -> None:
    _download_weights(force=force)
    download_ztf_records(ZTF_RECORDS_DIR, force=force)
