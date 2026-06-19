---
tags:
  - astronomy
  - time-series
  - light-curves
  - onnx
library_name: onnx
---

# ATAT

> Part of the **[light-curve](https://github.com/light-curve)** family of open-source tools for astronomical time-series analysis.
> Available from Python via the [`light-curve`](https://light-curve.snad.space/) package — `pip install light-curve`.
> Documentation: <https://light-curve.snad.space/>

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

Apache-2.0 (Copyright 2026 ALeRCE Collaboration) — matching the upstream
[alercebroker/ATAT](https://github.com/alercebroker/ATAT/blob/main/LICENSE)
repository, which adopted a license following
[alercebroker/ATAT#2](https://github.com/alercebroker/ATAT/issues/2). See
[LICENSE](LICENSE).

## Model overview

ATAT is a Transformer-based encoder for irregularly-sampled, multi-band astronomical light curves. The light-curve branch processes all six photometric bands jointly: each band's observations are independently embedded via a learned time modulation (sinusoidal Fourier basis), then all bands are merged, sorted by observation time, and passed through a 3-layer multi-head self-attention transformer. A learnable CLS token is prepended; its output at position 0 is the default representation. ATAT was trained for transient classification on the ELAsTiCC simulation (20 classes, LSST-like photometry).

Default configuration: 3 attention layers, 4 heads, head dimension 48 (d_model = 192), up to 65 observations per band, 6 bands, embedding dimension 192.

## Input data format

The model was trained on the ELAsTiCC simulation dataset which emulates LSST photometry in 6 bands (u, g, r, i, z, Y). Each light curve is represented as a multi-band time series of flux measurements.

## Preprocessing steps

Prepare inputs using the same pipeline applied during training (`get_lc_md.py`):

1. **Split by band.** Separate observations into 6 per-band sequences in the order `[u, g, r, i, z, Y]`. Each band gets an independent sequence of `(time, flux)` pairs.

2. **Pad or downsample to 65 per band.** If a band has fewer than 65 observations, right-pad with zeros to length 65. If a band has more than 65 observations, downsample to 65 by selecting indices `linspace(0, n−1, 65)` (uniform subsampling, not truncation). The result is `data [65, 6]` and `time [65, 6]`; padding positions hold `0.0`.

3. **Set the mask.** Set `mask = 1` for every slot containing a real observation and `mask = 0` for every padding slot. This is a direct validity indicator — set it based on how many real observations each band has, not derived from flux values.

4. **Shift time to start at zero.** Subtract the minimum observed time (across all bands, ignoring padding slots) from all valid time entries. Padding time slots remain `0.0`. Supply time in **days**; the model's internal time modulator uses `T_max = 1500` days, so it is calibrated for light curves spanning up to roughly four years.

5. **No flux normalisation.** Pass raw flux values without any normalisation. The model was trained on SNANA FLUXCAL with reference zero point ZP = 27.5 (a source at 27.5 AB mag has FLUXCAL = 1). Inputs from a different photometric system or flux scale are outside the training distribution and may produce poor embeddings.

## Inputs (ONNX)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `data` | `[batch, 65, 6]` | Per-band flux, SNANA FLUXCAL (ZP = 27.5), no normalisation |
| `time` | `[batch, 65, 6]` | Per-band observation times in days, shifted so earliest valid observation = 0; padding slots = 0 |
| `mask` | `[batch, 65, 6]` | **1 = valid observation, 0 = padding** |

## Outputs (ONNX)

Single file `atat.onnx` with three named outputs:

| Output name | Shape | Description |
|-------------|-------|-------------|
| `token`    | `[batch, 192]` | CLS token at position 0 after transformer (used in the paper) |
| `mean`     | `[batch, 192]` | Masked mean pooling over per-observation features |
| `sequence` | `[batch, 65×6, 192]` | Per-observation features (CLS token excluded) |

Request only the output(s) you need via `session.run(["token"], feed)` — onnxruntime will prune unused computation.

ONNX opset: 13.

## Weights

Source: [Google Drive — results\_paper.zip](https://drive.google.com/drive/folders/1uVOSJ1WMJH3o-5Czqx0WBcdmtf548GQ1?usp=sharing)
Training dataset: ELAsTiCC (DESC LSST simulation, 20 transient classes)
Checkpoint: `results/lc/Exp_cfg_-arch=lc-seed=0/`
