"""Export: copy the pre-built AstraCLR ONNX to the output directory."""

import shutil
from pathlib import Path

from astra_clr_prep.config import HF_ONNX_FILENAME, OUTPUT_FILENAME, WEIGHTS_DIR


def run_export(output_dir: Path) -> None:
    src = WEIGHTS_DIR / HF_ONNX_FILENAME
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run 'prep-models astra-clr download' first."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / OUTPUT_FILENAME
    shutil.copy2(src, dest)
    print(f"Copied ONNX to {dest}")
