"""Astromer 1 (ZTF DR20 g-band) preprocessing pipeline for test data generation.

Architecture and preprocessing are identical to the MACHO astromer1: the same
core/data.py load_numpy call with sampling=False, msk_frac=0, truncated to
max_obs=200 observations per light curve.
"""

from __future__ import annotations

import numpy as np
from prep_models_utils.astromer.common import AstromerConfig, add_code_to_path

_WINDOW_SIZE = 200


def preprocess_curves(config: AstromerConfig, curves: list[dict]) -> dict:
    add_code_to_path(config.code_dir)
    from core.data import load_numpy

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
        return {k: v.numpy() for k, v in batch.items()}
    raise RuntimeError("Empty dataset after preprocessing")
