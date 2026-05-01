from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AstromerConfig:
    model_name: str
    code_dir: Path
    weights_dir: Path
    zenodo_record_id: str
    zenodo_key: str
    output_prefix: str
    test_data_filename: str


_WINDOW_SIZE = 200


def add_code_to_path(code_dir: Path) -> None:
    code = str(code_dir)
    if code not in sys.path:
        sys.path.insert(0, code)


def run_encoder_mean(encoder, batch: dict) -> np.ndarray:
    """Run encoder and compute masked mean pooling.

    batch['mask_in'] uses the internal pipeline convention: 0=visible, 1=hidden/padding.
    The encoder receives it as-is (its attention uses 0=attend, 1=ignore).
    Pooling uses the complement (1-mask_in) to sum over valid timesteps only.
    This matches model_design.py: mask = 1. - mask_in.
    """
    import tensorflow as tf

    inputs = tf.constant(batch["input"], dtype=tf.float32)
    times = tf.constant(batch["times"], dtype=tf.float32)
    mask_in = tf.constant(batch["mask_in"], dtype=tf.float32)

    z = encoder({"input": inputs, "times": times, "mask_in": mask_in}, training=False)
    pooling_mask = 1.0 - mask_in
    return tf.math.divide_no_nan(
        tf.reduce_sum(z * pooling_mask, axis=1),
        tf.reduce_sum(pooling_mask, axis=1),
    ).numpy()


def download_weights(config: AstromerConfig) -> None:
    from prep_models_utils.zenodo import download_zenodo_file

    download_zenodo_file(
        config.zenodo_record_id,
        config.zenodo_key,
        config.weights_dir,
        extract_zip=True,
    )
    print(f"Weights saved to {config.weights_dir}")
