"""Download Astromer 1 pretrained weights and survey records."""

from prep_models_utils.astromer import download_weights
from prep_models_utils.astromer.surveys import (
    ALCOCK_GDRIVE_ID,
    ATLAS_GDRIVE_ID,
    download_records,
)

from astromer1_prep.config import CONFIG
from astromer1_prep.test_data import ALCOCK_RECORDS_DIR, ATLAS_RECORDS_DIR


def run_download(*, force: bool = False) -> None:
    download_weights(CONFIG)
    download_records(ALCOCK_GDRIVE_ID, ALCOCK_RECORDS_DIR, "alcock", force=force)
    download_records(ATLAS_GDRIVE_ID, ATLAS_RECORDS_DIR, "atlas", force=force)
