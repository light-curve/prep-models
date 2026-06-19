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

> Part of the **[light-curve](https://github.com/light-curve)** family of open-source tools for astronomical time-series analysis.
> Available from Python via the [`light-curve`](https://light-curve.snad.space/) package — `pip install light-curve`.
> Documentation: <https://light-curve.snad.space/>

**HuggingFace:** [light-curve/astromer1](https://huggingface.co/light-curve/astromer1)

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

All tensors are `float32`. Both magnitudes and times are **zero-mean normalized** before
passing to the model (subtract the per-window mean of each).

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input` | `[batch, 200, 1]` | `mag − mean(mag)` over the window |
| `times` | `[batch, 200, 1]` | `time − mean(time)` over the window |
| `mask_in` | `[batch, 200, 1]` | 1 = valid observation, 0 = padded position |

## Outputs (ONNX)

Single file `astromer1.onnx` with three named outputs:

| Output name | Shape | Aggregation |
|-------------|-------|-------------|
| `mean` | `[batch, 256]` | Masked mean pooling over valid timesteps |
| `max`  | `[batch, 256]` | Masked max pooling over valid timesteps |
| `sequence` | `[batch, 200, 256]` | Per-timestep features |

Request only the output(s) you need via `session.run(["mean"], feed)` — onnxruntime will prune unused computation.

ONNX opset: 13.

## Preprocessing steps

Photometric errors are **not used** at inference — only time and magnitude are needed.
The upstream code internally expects a 3-column `[time, mag, err]` array, but the error
column is dead code in the encoder (extracted but never used). Pass dummy zeros if
running the pipeline directly.

1. **Collect** observation times (in days — need not be absolute MJD) and magnitudes.
2. **Truncate** each light curve to at most 200 observations (take the first 200 if longer).
3. **Zero-mean normalize** both columns over the window:
   `time -= time.mean()`, `mag -= mag.mean()`
4. **Pad** shorter light curves to exactly 200 positions: append zeros to both `input` and `times`.
5. **Build the mask**: set `mask_in = 1` for real observations, `mask_in = 0` for padded positions.
6. **Reshape** each tensor to `[batch, 200, 1]` (add trailing dimension).

The sequence length is fixed at 200 by the pretrained weights.

## Weights

Source: [Zenodo record 18207945](https://zenodo.org/records/18207945)
Training dataset: MACHO R-band light curves
Checkpoint: `pt_macho_v1_2021.zip`

The test-data parquet file was generated with these MACHO weights using truncation to the first 200 observations.
