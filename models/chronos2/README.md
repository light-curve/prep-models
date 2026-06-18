# Chronos 2

## Paper

Ansari et al., 2025. "Chronos-2: Learning the Language of Time Series." Amazon Web Services.

```bibtex
@misc{ansari2025chronos2,
  title   = {Chronos-2: Learning the Language of Time Series},
  author  = {Abdul Fatir Ansari and others},
  year    = {2025},
  url     = {https://huggingface.co/amazon/chronos-2},
}
```

## Original model

HuggingFace: https://huggingface.co/amazon/chronos-2

Package: `chronos-forecasting>=2.0` (pip-installable, no code submodule)

## License

Apache 2.0

## Model overview

Chronos 2 is a 120 M-parameter encoder-only time series foundation model trained on a
large corpus of real and synthetic time series data.  It maps a sequence of scalar
observations into a sequence of patch embeddings via a T5-style transformer encoder.
The architecture uses non-overlapping patches of size 16 and instance normalisation
with arcsinh transformation.

## Irregular-sampling strategy

Following the [StarEmbed benchmark](https://arxiv.org/abs/2510.06200), timestamps are
**not** passed to Chronos 2.  The model encodes relative position implicitly via
sequential patch indices (scaled by the context length), so light curves are treated as
equally spaced in observation order.  Left-padding with NaN marks unused context
positions.

## Passband encoding

Chronos 2's `embed()` API accepts only magnitude values; there is **no built-in
passband channel**.  Multi-band data can be embedded per-band separately, or stacked as
multiple variates (shape `[n_bands, seq_len]`).  Explicit `lg(λ)` injection would
require direct calls to `model.encode()` with a custom input tensor — the architecture
has no dedicated wavelength embedding layer.

## Inputs

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `context` | `[batch, 512]` | float32 | Magnitude values; NaN marks left-padded positions |

The ONNX model has a fixed context length of 512 observations (32 patches × 16).
Longer series are truncated to the last 512 observations; shorter series are left-padded
with NaN.

## Outputs (ONNX)

| Name | Shape | Description |
|------|-------|-------------|
| `mean` | `[batch, 768]` | Masked mean pool over valid context patches |
| `sequence` | `[batch, 32, 768]` | Per-patch encoder hidden states |

## Preprocessing steps

1. Select observations (time-sorted magnitudes).
2. If more than 512 observations, keep the last 512.
3. Left-pad to length 512 with NaN.
4. Pass the `[batch, 512]` float32 tensor to the ONNX model.

Instance normalisation (mean subtraction, std scaling, arcsinh) is applied internally
by the model.

## Weights

Source: https://huggingface.co/amazon/chronos-2 (loaded automatically via
`Chronos2Pipeline.from_pretrained`)
