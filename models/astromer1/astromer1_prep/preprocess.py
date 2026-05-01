"""Astromer 1 preprocessing pipeline for test data generation.

Astromer 1's code lives under core/ (not src/), and load_numpy handles windowing,
normalisation, masking, and batching in a single call. For test data we want one
embedding per original light curve, so we truncate each curve to max_obs=200
observations before calling load_numpy (the same first-window behaviour as the
--sampling=False default in the original training scripts).
"""

from __future__ import annotations

import numpy as np

from prep_models_utils.astromer.common import AstromerConfig, add_code_to_path

_WINDOW_SIZE = 200


def preprocess_curves(config: AstromerConfig, curves: list[dict]) -> dict:
    """Process curves through the upstream Astromer 1 pipeline.

    Uses core/data.py: load_numpy with sampling=False, msk_frac=0.
    Curves are truncated to _WINDOW_SIZE before being passed so that
    get_windows produces exactly one window per light curve.

    mask_in semantics (upstream convention):
        0 = visible/real observation
        1 = masked or padding (excluded from pooling)
    """
    add_code_to_path(config.code_dir)
    from core.data import load_numpy

    # Truncate to window size so flat_map produces exactly one window per LC
    samples = [
        np.stack(
            [
                np.asarray(c["mjd"], dtype=np.float32),
                np.asarray(c["magnitude"], dtype=np.float32),
                np.asarray(c["error"], dtype=np.float32),
            ],
            axis=1,
        )[:_WINDOW_SIZE]
        for c in curves
    ]

    dataset = load_numpy(
        samples,
        batch_size=len(curves),
        max_obs=_WINDOW_SIZE,
        msk_frac=0.0,
        rnd_frac=0.0,
        same_frac=0.0,
        sampling=False,
    )

    for batch in dataset:
        # mask_sample returns a dict; after padded_batch this is a batched dict
        return {k: v.numpy() for k, v in batch.items()}
    raise RuntimeError("Empty dataset after preprocessing")
