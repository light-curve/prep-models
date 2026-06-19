"""Generate test data: synthetic periodic light curves + Chronos 2 embeddings.

Curve generation, embedding, and parquet writing are shared across the
Chronos family in ``prep_models_utils.chronos``; this module only supplies the
Chronos 2 embedder and its native context length.
"""

from __future__ import annotations

from pathlib import Path

from prep_models_utils.chronos import run_test_data as run_shared_test_data

from chronos2_prep.config import OUTPUT_PREFIX
from chronos2_prep.export import _Chronos2Embedder, _load_pipeline

# Chronos 2's native context length.
_MAX_OBS = 8192


def run_test_data(output_dir: Path, n_samples: int = 10) -> None:
    pipeline = _load_pipeline()
    pipeline.model.eval()
    embedder = _Chronos2Embedder(pipeline)
    run_shared_test_data(
        output_dir,
        n_samples,
        embedder=embedder,
        d_model=pipeline.model.model_dim,
        output_prefix=OUTPUT_PREFIX,
        max_obs=_MAX_OBS,
    )
