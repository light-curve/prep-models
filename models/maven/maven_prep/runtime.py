"""Build the Maven light-curve encoder and load the fine-tuned weights."""

from __future__ import annotations

import sys

import torch
from torch import nn

from maven_prep.config import (
    AGG,
    CHECKPOINT_NAME,
    CODE_DIR,
    EMB,
    ENC_DIM,
    HEADS,
    N_OUT,
    NBAND,
    TIME_NORM,
    TRANSFORMER_DEPTH,
    WEIGHTS_DIR,
)


def add_code_to_path() -> None:
    """Make upstream `src` importable.

    Only `src.transformer_utils` is imported (pure torch); `src.models_multimodal`
    pulls in pytorch-lightning, wandb and friends, which we avoid.
    """
    p = str(CODE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


class MavenLightCurveEncoder(nn.Module):
    """The light-curve branch of upstream `LightCurveImageCLIP`.

    Mirrors `LightCurveImageCLIP.lightcurve_embeddings_with_projection`:
    `lightcurve_encoder` (transformer + masked mean + projection to n_out)
    followed by `lightcurve_projection` (n_out -> enc_dim) and L2 normalization.
    """

    def __init__(self) -> None:
        super().__init__()
        add_code_to_path()
        from src.transformer_utils import TransformerWithTimeEmbeddings

        self.lightcurve_encoder = TransformerWithTimeEmbeddings(
            n_out=N_OUT,
            nband=NBAND,
            agg=AGG,
            time_norm=TIME_NORM,
            emb=EMB,
            heads=HEADS,
            depth=TRANSFORMER_DEPTH,
            dropout=0.0,
        )
        self.lightcurve_projection = nn.Linear(N_OUT, ENC_DIM)

    def forward(
        self, x_lc: torch.Tensor, t_lc: torch.Tensor, mask_lc: torch.Tensor
    ) -> torch.Tensor:
        x = x_lc[..., None]
        x = self.lightcurve_encoder(x, t_lc, mask_lc)
        x = self.lightcurve_projection(x)
        return x / x.norm(dim=-1, keepdim=True)


def load_model() -> MavenLightCurveEncoder:
    path = WEIGHTS_DIR / CHECKPOINT_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Weights not found at {path}. Run 'prep-models maven download' first."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = {
        k: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(("lightcurve_encoder.", "lightcurve_projection."))
    }
    model = MavenLightCurveEncoder()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
