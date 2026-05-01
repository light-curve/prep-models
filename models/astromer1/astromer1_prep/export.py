"""Export Astromer 1 encoder to ONNX (one file per aggregation variant)."""

import json
from pathlib import Path

from prep_models_utils.astromer import run_export as run_shared_export
from prep_models_utils.astromer.common import add_code_to_path

from astromer1_prep.config import CODE_DIR, CONFIG, WEIGHTS_DIR


def _find_weights_dir() -> Path:
    if not (WEIGHTS_DIR / "conf.json").exists():
        raise FileNotFoundError(
            f"conf.json not found in {WEIGHTS_DIR}. "
            "Run 'prep-models astromer1 download' first."
        )
    return WEIGHTS_DIR


def _load_model():
    """Load Astromer 1 model by building the Encoder directly.

    get_ASTROMER() fails under Keras 3 because it traces EncoderLayer.call()
    symbolically and the layer passes `training` as a positional arg.
    We build the Encoder layer directly with a dummy input to assign weights,
    then wrap it in a minimal Model so the rest of the pipeline (which calls
    model.get_layer("encoder")) still works.
    """
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

    # Wrap encoder in a Model so load_weights is available
    inp = tf.keras.Input(shape=(max_obs, 1), name="input")
    tms = tf.keras.Input(shape=(max_obs, 1), name="times")
    msk = tf.keras.Input(shape=(max_obs, 1), name="mask_in")
    out = encoder({"input": inp, "times": tms, "mask_in": msk}, training=False)
    model = tf.keras.Model(
        inputs={"input": inp, "times": tms, "mask_in": msk},
        outputs=out,
        name="ASTROMER",
    )

    # The full checkpoint includes a regression head; expect_partial() silently
    # skips weights not present in this encoder-only model.
    model.load_weights(str(weights_dir / "weights")).expect_partial()

    return model, conf


def run_export(output_dir: Path) -> None:
    run_shared_export(output_dir, config=CONFIG, load_model=_load_model)
