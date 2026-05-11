from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_SRC = _REPO_ROOT / "code" / "src"
_WEIGHTS_DIR = _REPO_ROOT / "weights"


def add_code_to_path() -> None:
    """Add upstream src/ to sys.path so `import informer` and `import model` work."""
    p = str(_CODE_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


def weights_file() -> Path:
    return _WEIGHTS_DIR / "model.safetensors"


def load_model():
    """Load the photo encoder (Informer, classification=False) with pretrained weights."""
    from safetensors.torch import load_file

    add_code_to_path()
    from model import Informer
    from astrom3_prep.config import (
        ACTIVATION,
        D_FF,
        D_MODEL,
        DROPOUT,
        E_LAYERS,
        ENC_IN,
        FACTOR,
        N_HEADS,
        SEQ_LEN,
    )

    informer = Informer(
        classification=False,
        num_classes=10,  # unused when classification=False
        seq_len=SEQ_LEN,
        enc_in=ENC_IN,
        d_model=D_MODEL,
        dropout=DROPOUT,
        factor=FACTOR,
        output_attention=False,
        n_heads=N_HEADS,
        d_ff=D_FF,
        activation=ACTIVATION,
        e_layers=E_LAYERS,
    )

    wf = weights_file()
    if not wf.exists():
        raise FileNotFoundError(
            f"Weights not found at {wf}. Run 'prep-models astrom3 download' first."
        )
    weights = load_file(str(wf))
    # Load all weights; strict=False so fc.* (classification head) is silently ignored
    informer.load_state_dict(weights, strict=False)
    informer.eval()
    return informer
