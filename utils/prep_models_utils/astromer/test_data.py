from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from prep_models_utils.parquet import save_test_data

from .common import AstromerConfig, run_encoder_mean


def run_test_data(
    output_dir: Path,
    n_samples: int,
    *,
    config: AstromerConfig,
    load_model: Callable[[], tuple[object, object]],
    load_curves: Callable[[int], list[dict]],
    preprocess_fn: Callable | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading light curves ...")
    curves = load_curves(n_samples)

    print("Applying original Astromer preprocessing ...")
    if preprocess_fn is None:
        raise ValueError("preprocess_fn is required")
    _preprocess = preprocess_fn
    batch = _preprocess(config, curves)

    print("Loading model and computing embeddings ...")
    model, _ = load_model()
    encoder = model.get_layer("encoder")
    embeddings = run_encoder_mean(encoder, batch)

    rows = []
    for idx, curve in enumerate(curves):
        row = {}
        for k in ("lcid", "label", "survey"):
            if k in curve:
                v = curve[k]
                row[k] = v.item() if hasattr(v, "item") else v
        mjd = np.asarray(curve["mjd"]).tolist()
        mag = np.asarray(curve["magnitude"]).tolist()
        err = np.asarray(curve["error"]).tolist()
        row["lightcurve"] = [
            {"mjd": t, "mag": m, "magerr": e} for t, m, e in zip(mjd, mag, err)
        ]
        row["embedding_mean"] = embeddings[idx].tolist()
        rows.append(row)

    out_path = output_dir / config.test_data_filename
    save_test_data(rows, out_path)
    print(f"Saved {len(rows)} samples to {out_path}")
