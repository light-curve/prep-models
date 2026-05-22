"""Export: copy the pre-built AstraCLR ONNX and rename output → mean."""

from pathlib import Path

import onnx

from astra_clr_prep.config import HF_ONNX_FILENAME, OUTPUT_FILENAME, WEIGHTS_DIR

_OLD_NAME = "output"
_NEW_NAME = "mean"


def _rename_output(model: onnx.ModelProto) -> onnx.ModelProto:
    for out in model.graph.output:
        if out.name == _OLD_NAME:
            out.name = _NEW_NAME
    for node in model.graph.node:
        node.output[:] = [_NEW_NAME if o == _OLD_NAME else o for o in node.output]
    return model


def run_export(output_dir: Path) -> None:
    src = WEIGHTS_DIR / HF_ONNX_FILENAME
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run 'prep-models astra-clr download' first."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / OUTPUT_FILENAME
    model = onnx.load(str(src))
    model = _rename_output(model)
    onnx.save(model, str(dest))
    print(f"Exported ONNX to {dest}")
