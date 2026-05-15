"""Export Astromer 1 (ZTF DR20 g-band) encoder to ONNX.

The weights come from the ANN_clf checkpoint (full FCATT model: encoder +
FC classification head) saved by Nakoneczny et al. 2025.  We load with
expect_partial() so the FC head weights are silently ignored, leaving only
the encoder — identical to the approach used for the MACHO astromer1 weights.
"""

import json
from pathlib import Path

from prep_models_utils.astromer import run_export as run_shared_export
from prep_models_utils.astromer.common import add_code_to_path

from astromer1_ztfdr20_prep.config import CODE_DIR, CONFIG, WEIGHTS_DIR


def _find_weights_dir() -> Path:
    if not (WEIGHTS_DIR / "conf.json").exists():
        raise FileNotFoundError(
            f"conf.json not found in {WEIGHTS_DIR}. "
            "Run 'prep-models astromer1-ztfdr20 download' first."
        )
    return WEIGHTS_DIR


def _load_model():
    """Load the ZTF DR20 g-band encoder by extracting it from the FCATT checkpoint."""
    add_code_to_path(CODE_DIR)
    import tensorflow as tf
    from core.encoder import Encoder

    weights_dir = _find_weights_dir()
    with open(weights_dir / "conf.json") as f:
        conf = json.load(f)

    max_obs = conf["max_obs"]

    encoder = Encoder(
        num_layers=conf["layers"],
        d_model=conf["head_dim"],
        num_heads=conf["heads"],
        dff=conf["dff"],
        base=conf["base"],
        rate=conf["dropout"],
        use_leak=conf["use_leak"],
        name="encoder",
    )

    inp = tf.keras.Input(shape=(max_obs, 1), name="input")
    tms = tf.keras.Input(shape=(max_obs, 1), name="times")
    msk = tf.keras.Input(shape=(max_obs, 1), name="mask_in")
    out = encoder({"input": inp, "times": tms, "mask_in": msk}, training=False)
    model = tf.keras.Model(
        inputs={"input": inp, "times": tms, "mask_in": msk},
        outputs=out,
        name="ASTROMER",
    )

    # The checkpoint includes FC classification head weights; expect_partial()
    # silently skips them, loading only the encoder (layer_with_weights-0).
    model.load_weights(str(weights_dir / "weights")).expect_partial()

    return model, conf


def run_export(output_dir: Path) -> None:
    run_shared_export(output_dir, config=CONFIG, load_model=_load_model)
