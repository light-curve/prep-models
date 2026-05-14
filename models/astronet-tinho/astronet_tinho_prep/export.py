"""Export AstroNet-TINHO to ONNX via tf2onnx.

TINHO is a compressed (weight-clustered) variant of the T2 time-series
transformer, trained on PLAsTiCC to classify 14 types of astronomical
transients.  The upstream repository ships the trained SavedModels directly,
so no separate weight download is needed.

Two SavedModels are exported:

  tinho_gr.onnx      – g+r bands only, no redshift
                       input:  flux (B, 100, 2)  float32
                       output: probs (B, 14)      float32

  tinho_ugrizy.onnx  – all 6 LSST bands + host redshift
                       inputs: flux (B, 100, 6)  float32
                               redshift (B, 2)   float32
                       output: probs (B, 14)      float32

Output index → PLAsTiCC class mapping:
  0:SNIa  1:SNIa-91bg  2:SNIax  3:SNII  4:SNIbc  5:SLSN-I  6:TDE
  7:KN  8:AGN  9:RRL  10:M-dwarf  11:EB  12:Mira  13:μ-Lens-Single
(class order is fixed by the label encoder used during training)
"""

from __future__ import annotations

from pathlib import Path


_CODE_DIR = Path(__file__).resolve().parent.parent / "code"
_MODELS_DIR = _CODE_DIR / "astronet" / "tinho" / "models" / "plasticc"

# Model directory name prefixes; we glob for the exact name
_GR_PREFIX = "model-GR-noZ-28341-"
_UGRIZY_PREFIX = "model-UGRIZY-wZ-31367-"


def _find_model(prefix: str) -> Path:
    matches = [
        p for p in _MODELS_DIR.iterdir() if p.name.startswith(prefix) and p.is_dir()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No SavedModel matching '{prefix}*' in {_MODELS_DIR}. "
            "Ensure the git submodule (models/astronet-tinho/code) is initialised."
        )
    return matches[0]


def run_export(out_dir: Path) -> None:
    import tf2onnx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for prefix, out_name in [
        (_GR_PREFIX, "tinho_gr.onnx"),
        (_UGRIZY_PREFIX, "tinho_ugrizy.onnx"),
    ]:
        saved_model_path = _find_model(prefix)
        out_path = out_dir / out_name

        print(f"Converting {saved_model_path.name} → {out_path} …")
        tf2onnx.convert.from_saved_model(
            str(saved_model_path),
            output_path=str(out_path),
            opset=17,
        )
        print(f"  Written: {out_path}")

    print(f"\nAll ONNX files written to {out_dir}")
