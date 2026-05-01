---
tags:
  - astronomy
  - time-series
  - light-curves
  - onnx
library_name: onnx
---

# ATAT

## Paper

Becker, I., Pignata, G., Förster, F., Estévez, P. A., Cabrera-Vives, G., Vera, E., Carrasco-Davis, R., Astorga, N., Sanchez-Saez, P., Catelan, M., Cortés, C. C., de Jaeger, T., Pezoa, F., & Reyes, I. (2024). *ATAT: Astronomical Transformer for time series And Tabular data*. Astronomy & Astrophysics, 691, A163.

```bibtex
@article{atat2024,
  author  = {Becker, I. and Pignata, G. and F{\"{o}}rster, F. and
             Est{\'e}vez, P. A. and Cabrera-Vives, G. and Vera, E. and
             Carrasco-Davis, R. and Astorga, N. and Sanchez-Saez, P. and
             Catelan, M. and Cort{\'e}s, C. C. and {de Jaeger}, T. and
             Pezoa, F. and Reyes, I.},
  title   = {{ATAT}: Astronomical Transformer for time series And Tabular data},
  journal = {Astronomy \& Astrophysics},
  year    = {2024},
  volume  = {691},
  pages   = {A163},
  doi     = {10.1051/0004-6361/202451418},
}
```

## Original code

<https://github.com/alercebroker/ATAT> (git submodule at `models/atat/code/`)

## License

The original ATAT repository has no license as of the time of writing (see [alercebroker/ATAT#2](https://github.com/alercebroker/ATAT/issues/2)).
Until a license is added upstream, the ONNX models cannot be published to HuggingFace.

## Model overview

ATAT is a Transformer-based encoder for irregularly-sampled, multi-band astronomical light curves. The light-curve branch processes all six photometric bands jointly: each band's observations are independently embedded via a learned time modulation (sinusoidal Fourier basis), then all bands are merged, sorted by observation time, and passed through a 3-layer multi-head self-attention transformer. A learnable CLS token is prepended; its output at position 0 is the default representation. ATAT was trained for transient classification on the ELAsTiCC simulation (20 classes, LSST-like photometry).

Default configuration: 3 attention layers, 4 heads, head dimension 48 (d_model = 192), up to 65 observations per band, 6 bands, embedding dimension 192.

## Input data format

The model was trained on the ELAsTiCC simulation dataset which emulates LSST photometry in 6 bands (u, g, r, i, z, Y). Each light curve is represented as a multi-band time series of flux measurements.

## Preprocessing steps

The ELASTICC data pipeline (in `datasets.py`) applies the following steps before passing data to the encoder:

1. Load multi-band light curve from `.h5` file: `data [seq_len, 6]`, `time [seq_len, 6]`, `mask [seq_len, 6]`.
2. Mask out outlier observations (flux > median ± 5σ, or error > median ± 5σ per band).
3. Right-pad each band's sequence to `seq_len = 65` with zeros; set `mask = 0` for padding positions.
4. No global normalisation of flux or time is applied in the data loader — the time modulator handles time scaling internally via `T_max = 1500`.

For inference with the ONNX model, prepare inputs accordingly:
- `data`: per-band flux values, `float32`, shape `[batch, 65, 6]`
- `time`: per-band observation times (same units as training, 0–1500 range), `float32`, shape `[batch, 65, 6]`
- `mask`: `1` for real observations, `0` for padding, `float32`, shape `[batch, 65, 6]`

Bands are ordered `[u, g, r, i, z, Y]` (indices 0–5).

## Inputs (ONNX)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `data` | `[batch, 65, 6]` | Per-band flux (no global normalisation) |
| `time` | `[batch, 65, 6]` | Per-band observation times (0–1500) |
| `mask` | `[batch, 65, 6]` | **1 = valid observation, 0 = padding** |

The ATAT internal convention uses the same mask sign (1=valid), so no inversion is applied.

## Outputs (ONNX)

| File | Output shape | Aggregation |
|------|-------------|-------------|
| `atat_token.onnx` | `[batch, 192]` | CLS token at position 0 after transformer (used in the paper) |
| `atat_mean.onnx`  | `[batch, 192]` | Masked mean pooling over sequence positions |
| `atat_full.onnx`  | `[batch, 1+65×6, 192]` | Full sequence including CLS token at position 0 |

ONNX opset: 13.

## Weights

Source: [Google Drive — results\_paper.zip](https://drive.google.com/drive/folders/1uVOSJ1WMJH3o-5Czqx0WBcdmtf548GQ1?usp=sharing)
Training dataset: ELAsTiCC (DESC LSST simulation, 20 transient classes)
Checkpoint: `results/lc/Exp_cfg_-arch=lc-seed=0/`
