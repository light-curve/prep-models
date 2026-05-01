---
license: mit
tags:
  - astronomy
  - time-series
  - light-curves
  - onnx
library_name: onnx
---

# Astromer 1

## Paper

Donoso-Oliva, C., Becker, I., Protopapas, P., Cabrera-Vives, G., Forster, F., & Estévez, P. A. (2023). *ASTROMER: A transformer-based embedding for the representation of light curves*. Astronomy & Astrophysics, 670, A54.

```bibtex
@article{astromer1,
  author  = {Donoso-Oliva, C. and Becker, I. and Protopapas, P. and
             Cabrera-Vives, G. and Forster, F. and Est{\'e}vez, P. A.},
  title   = {{ASTROMER}: A transformer-based embedding for the representation
             of light curves},
  journal = {Astronomy \& Astrophysics},
  volume  = {670},
  pages   = {A54},
  year    = {2023},
  doi     = {10.1051/0004-6361/202243928},
}
```

## Original code

<https://github.com/astromer-science/main-code> (Astromer v1 tag)

## License

MIT — see [LICENSE](LICENSE).

## Model overview

Astromer 1 is a transformer encoder pretrained on MACHO R-band light curves via
masked magnitude prediction. It maps irregularly-sampled photometric time series
to per-timestep contextual embeddings using an MJD-aware sinusoidal positional
encoding. The architecture uses 2 transformer layers, 4 attention heads, and a
head dimension of 64, producing 256-dimensional embeddings.

## Inputs

All tensors are `float32`. Magnitudes must be **zero-mean normalized** before
passing to the model (subtract the per-light-curve mean magnitude).

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input` | `[batch, 200, 1]` | Zero-mean normalized magnitudes |
| `times` | `[batch, 200, 1]` | Observation times in MJD |
| `mask_in` | `[batch, 200, 1]` | 1 = valid observation, 0 = padded position |

## Outputs (ONNX)

| File | Output shape | Aggregation |
|------|-------------|-------------|
| `astromer1_mean.onnx` | `[batch, 256]` | Masked mean pooling over valid timesteps |
| `astromer1_max.onnx` | `[batch, 256]` | Masked max pooling over valid timesteps |
| `astromer1_full.onnx` | `[batch, 200, 256]` | Full per-timestep sequence |

ONNX opset: 13.

## Preprocessing steps

1. **Collect** MJD observation times and magnitudes for each light curve.
2. **Zero-mean normalize** magnitudes: subtract the mean magnitude of each light curve individually (`mag -= mag.mean()`).
3. **Truncate** each light curve to at most 200 observations (take the first 200 if longer).
4. **Pad** shorter light curves to exactly 200 positions: append zeros to both `input` and `times`.
5. **Build the mask**: set `mask_in = 1` for real observations, `mask_in = 0` for padded positions.
6. **Reshape** each tensor to `[batch, 200, 1]` (add trailing dimension).

The sequence length is fixed at 200 by the pretrained weights.

## Weights

Source: [Zenodo record 18207945](https://zenodo.org/records/18207945)
Training dataset: MACHO R-band light curves
