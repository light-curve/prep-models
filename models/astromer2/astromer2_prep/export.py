"""Export Astromer 2 encoder to ONNX (one file per aggregation variant)."""

from pathlib import Path

from prep_models_utils.astromer import run_export as run_shared_export

from astromer2_prep.config import CODE_DIR, CONFIG, WEIGHTS_DIR
from astromer2_prep.preprocess import load_model_from_checkpoint


def _find_weights_dir() -> Path:
    config = WEIGHTS_DIR / "config.toml"
    if not config.exists():
        raise FileNotFoundError(
            f"config.toml not found in {WEIGHTS_DIR}. "
            "Run 'prep-models astromer2 download' first."
        )
    return WEIGHTS_DIR


def _load_model():
    return load_model_from_checkpoint(CODE_DIR, _find_weights_dir())


def run_export(output_dir: Path) -> None:
    run_shared_export(output_dir, config=CONFIG, load_model=_load_model)
