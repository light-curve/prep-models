# MOMENT-1

> Part of the **[light-curve](https://github.com/light-curve)** family of open-source tools for astronomical time-series analysis.
> Available from Python via the [`light-curve`](https://light-curve.snad.space/) package — `pip install light-curve`.
> Documentation: <https://light-curve.snad.space/>

## Paper

Goswami et al., 2024. "MOMENT: A Family of Open Time-series Foundation Models."
Proceedings of the 41st International Conference on Machine Learning (ICML).

```bibtex
@inproceedings{goswami2024moment,
  title     = {MOMENT: A Family of Open Time-series Foundation Models},
  author    = {Goswami, Mononito and Szafer, Konrad and Choudhry, Arjun and Cai, Yifu and Li, Shuo and Dubrawski, Artur},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024},
}
```

## Original model

HuggingFace:
[small](https://huggingface.co/AutonLab/MOMENT-1-small) ·
[base](https://huggingface.co/AutonLab/MOMENT-1-base) ·
[large](https://huggingface.co/AutonLab/MOMENT-1-large)

Package: `momentfm==0.1.4` (pip-installable, no code submodule)

GitHub: <https://github.com/moment-timeseries-foundation-model/moment>

Each size is pinned to a specific HF commit (see `moment_prep/config.py`) so the
export is fully reproducible regardless of upstream changes to `main`.

## License

MIT (Auton Lab, Carnegie Mellon University)

## Model overview

MOMENT is a family of T5-based time-series foundation models pre-trained with a
masked-reconstruction objective on the Time-series Pile.  The input is split
into non-overlapping patches of 8 observations, normalized with reversible
instance normalization (RevIN), embedded with a linear patch projection plus a
learned positional embedding, and encoded by a Flan-T5 encoder stack.  We export
the **encoder in "embedding" mode**: it produces one embedding per patch, which
is masked-mean-pooled over the valid patches to yield a single fixed-size vector.
The three sizes differ only in width:

| Size | Backbone | `d_model` |
|------|----------|-----------|
| small | Flan-T5-small | 512 |
| base  | Flan-T5-base  | 768 |
| large | Flan-T5-large | 1024 |

All three share an identical ONNX interface and preprocessing.

## Fixed context length

Unlike the Chronos family, MOMENT has a **fixed** context window of 512
observations (64 patches of 8); the sequence length is *not* a dynamic axis.
Cap each light curve to its last 512 observations and left-pad shorter curves
to 512 with NaN.

## Time / irregular-sampling strategy

MOMENT's architecture has **no timestamp input** — `embed()` takes only the
value tensor `x_enc` and a boolean `input_mask`; positional information is purely
index-based (a sinusoidal embedding over patch index), and the model was
pre-trained on regularly sampled series.  There is therefore no way to feed
modified Julian dates directly.

This export follows the same timestamp-free convention this repo uses for the
Chronos models: light curves are treated as equally spaced in observation order,
and NaN marks padded / missing positions.  (That convention comes from the
[StarEmbed benchmark](https://arxiv.org/abs/2510.06200), which evaluates Chronos
and Astromer but **not** MOMENT — we extend it to MOMENT for consistency.)

If you instead want the model to "see" real time, you must do it in
pre-processing: resample / bin the light curve onto a uniform 512-step time grid
so that patch index corresponds to elapsed time, marking empty bins with NaN.
The ONNX interface supports both approaches unchanged — only how you fill the
512-step `context` differs.

## Inputs

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `context` | `[batch, 512]` | float32 | Magnitude values; NaN marks padded / missing positions |

`batch` is **dynamic**; the sequence length is fixed at 512.  A patch of 8
observations contributes to the output only if all 8 of its positions are valid
(non-NaN), matching MOMENT's patch-level masking.

## Outputs (ONNX)

One file per size, `moment1-<size>.onnx`, with two named outputs:

| Name | Shape | Description |
|------|-------|-------------|
| `mean` | `[batch, d_model]` | Masked mean pool over valid patches (the embedding) |
| `sequence` | `[batch, 64, d_model]` | Per-patch encoder hidden states |

## Preprocessing steps

1. Select observations (time-sorted magnitudes).
2. Cap to the last 512 observations (MOMENT's native context).
3. Left-pad each series to 512 with NaN.
4. Pass the `[batch, 512]` float32 tensor to the ONNX model.

RevIN instance normalisation (masked mean subtraction, std scaling) is applied
internally by the model.

## Weights

Source: `AutonLab/MOMENT-1-{small,base,large}` (loaded automatically via
`MOMENTPipeline.from_pretrained`, each pinned to a specific revision).
Dataset: the Time-series Pile (`AutonLab/Timeseries-PILE`), a large collection
of public time series spanning many domains.
