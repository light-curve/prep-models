# Chronos-Bolt

> Part of the **[light-curve](https://github.com/light-curve)** family of open-source tools for astronomical time-series analysis.
> Available from Python via the [`light-curve`](https://light-curve.snad.space/) package — `pip install light-curve`.
> Documentation: <https://light-curve.snad.space/>

## Paper

Ansari et al., 2024. "Chronos: Learning the Language of Time Series." Transactions
on Machine Learning Research. Chronos-Bolt is a faster, patch-based variant
released by Amazon Web Services.

```bibtex
@article{ansari2024chronos,
  title   = {Chronos: Learning the Language of Time Series},
  author  = {Ansari, Abdul Fatir and others},
  journal = {Transactions on Machine Learning Research},
  year    = {2024},
  url     = {https://github.com/amazon-science/chronos-forecasting},
}
```

## Original model

HuggingFace:
[tiny](https://huggingface.co/amazon/chronos-bolt-tiny) ·
[mini](https://huggingface.co/amazon/chronos-bolt-mini) ·
[small](https://huggingface.co/amazon/chronos-bolt-small) ·
[base](https://huggingface.co/amazon/chronos-bolt-base)

Package: `chronos-forecasting==2.3.0` (pip-installable, no code submodule)

Each size is pinned to a specific HF commit (see `chronos_bolt_prep/config.py`)
so the export is fully reproducible regardless of upstream changes to `main`.

## License

Apache-2.0

## Model overview

Chronos-Bolt is an encoder–decoder time series foundation model.  Like Chronos 2,
it splits the input into non-overlapping patches of size 16, applies instance
normalization, and encodes the patches with a T5-style transformer.  We export
**only the encoder** to produce embeddings; the four sizes differ only in width:

| Size | `d_model` |
|------|-----------|
| tiny | 256 |
| mini | 384 |
| small | 512 |
| base | 768 |

All four share an identical ONNX interface and preprocessing — the same as the
Chronos 2 export.

## Irregular-sampling strategy

Following the [StarEmbed benchmark](https://arxiv.org/abs/2510.06200), timestamps
are **not** passed to the model.  Light curves are treated as equally spaced in
observation order; left-padding with NaN marks unused context positions.

## Inputs

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `context` | `[batch, seq]` | float32 | Magnitude values; NaN marks left-padded positions |

Both `batch` and `seq` are **dynamic** axes.  `seq` must be a multiple of the
patch size (16) and may be anything up to the model's native context of 2048
(128 patches).  Inference cost scales with `seq`, so shorter series are
proportionally cheaper — there is no fixed window and no truncation.  Pad each
batch to a common multiple of 16 with NaN.

## Outputs (ONNX)

One file per size, `chronos-bolt-<size>.onnx`, with two named outputs:

| Name | Shape | Description |
|------|-------|-------------|
| `mean` | `[batch, d_model]` | Masked mean pool over valid context patches |
| `sequence` | `[batch, seq/16, d_model]` | Per-patch encoder hidden states |

## Preprocessing steps

1. Select observations (time-sorted magnitudes).
2. Optionally cap to the last 2048 observations (the model's native context).
3. Left-pad each series to a common length that is a multiple of 16, using NaN.
4. Pass the `[batch, seq]` float32 tensor to the ONNX model.

Instance normalisation (mean subtraction, std scaling) is applied internally by
the model.

## Weights

Source: `amazon/chronos-bolt-{tiny,mini,small,base}` (loaded automatically via
`ChronosBoltPipeline.from_pretrained`, each pinned to a specific revision).
