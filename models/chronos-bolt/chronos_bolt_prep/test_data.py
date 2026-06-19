"""Generate test data: synthetic periodic light curves + Chronos-Bolt embeddings.

Curve generation, embedding, and parquet writing are shared across the
Chronos family in ``prep_models_utils.chronos``; this module only supplies the
Chronos-Bolt embedder (per size) and its native context length.
"""

from __future__ import annotations

from pathlib import Path

from prep_models_utils.chronos import run_test_data as run_shared_test_data

from chronos_bolt_prep.config import output_prefix
from chronos_bolt_prep.export import _ChronosBoltEmbedder, _load_pipeline

# Chronos-Bolt's native context length.
_MAX_OBS = 2048


def run_test_data(output_dir: Path, n_samples: int = 10, *, size: str) -> None:
    pipeline = _load_pipeline(size)
    pipeline.model.eval()
    embedder = _ChronosBoltEmbedder(pipeline)
    run_shared_test_data(
        output_dir,
        n_samples,
        embedder=embedder,
        d_model=pipeline.model.model_dim,
        output_prefix=output_prefix(size),
        max_obs=_MAX_OBS,
    )
