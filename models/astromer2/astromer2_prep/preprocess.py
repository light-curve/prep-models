"""Astromer 2 preprocessing pipeline for test data generation and ONNX export.

Uses the upstream src/data pipeline: load_numpy → to_windows → standardize →
mask_dataset (msk_frac=0) → padded_batch → format_inp_astromer.

mask_in semantics (upstream convention):
    0 = visible/real observation
    1 = masked or padding (excluded from pooling)
"""

from __future__ import annotations

import numpy as np

from prep_models_utils.astromer.common import AstromerConfig, add_code_to_path

_WINDOW_SIZE = 200


def preprocess_curves(config: AstromerConfig, curves: list[dict]) -> dict:
    add_code_to_path(config.code_dir)
    from src.data.loaders import load_numpy, format_inp_astromer
    from src.data import preprocessing as pp
    from src.data.masking import mask_dataset

    samples = [
        np.stack(
            [
                np.asarray(c["mjd"], dtype=np.float32),
                np.asarray(c["magnitude"], dtype=np.float32),
                np.asarray(c["error"], dtype=np.float32),
            ],
            axis=1,
        )
        for c in curves
    ]

    dataset = load_numpy(samples)
    dataset = pp.to_windows(dataset, window_size=_WINDOW_SIZE, sampling=False)
    dataset = dataset.map(pp.standardize)
    # msk_frac=0: no observations are hidden; padding positions get mask_in=1
    dataset, shapes = mask_dataset(
        dataset, msk_frac=0.0, rnd_frac=0.0, same_frac=0.0, window_size=_WINDOW_SIZE
    )
    dataset = dataset.padded_batch(len(curves), padded_shapes=shapes)
    dataset = dataset.map(lambda x: format_inp_astromer(x, aversion="base"))

    for inp, _ in dataset:
        return {k: v.numpy() for k, v in inp.items()}
    raise RuntimeError("Empty dataset after preprocessing")


def load_model_from_checkpoint(code_dir, checkpoint_dir):
    import toml

    add_code_to_path(code_dir)
    from presentation.pipelines.steps.model_design import build_model

    config = toml.load(checkpoint_dir / "config.toml")
    model = build_model(config)
    model.load_weights(str(checkpoint_dir / "weights")).expect_partial()
    return model, config
