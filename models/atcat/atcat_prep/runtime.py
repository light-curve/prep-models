from __future__ import annotations

import os
import sys
from pathlib import Path

import fiddle as fdl
import torch

from atcat_prep.config import (
    CHECKPOINT_PATH,
    CODE_DIR,
    CONFIG_JSON_PATH,
    DATA_DIR,
    WEIGHTS_DIR,
)


def add_code_to_path() -> None:
    code = str(CODE_DIR)
    if code not in sys.path:
        sys.path.insert(0, code)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def checkpoint_file() -> Path:
    return WEIGHTS_DIR / CHECKPOINT_PATH


def config_json_file() -> Path:
    return WEIGHTS_DIR / CONFIG_JSON_PATH


def test_parquet_dir() -> Path:
    return DATA_DIR / "data_parquet" / "split_0"


def load_experiment():
    add_code_to_path()
    from atcat.config import json_serialize_config
    from atcat.configs.flattened.core import lc_only_split_0

    config_path = config_json_file()
    if config_path.exists():
        config = json_serialize_config.loads(config_path.read_text())
    else:
        config = lc_only_split_0.config_fixture()
    return fdl.build(config)


def load_model() -> torch.nn.Module:
    exp = load_experiment()
    model = exp.architecture.make_model(exp)
    ckpt_path = checkpoint_file()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run 'prep-models atcat download' first."
        )
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected state_dict mismatch. Missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model
