---
license: mit
tags:
  - astronomy
  - time-series
  - light-curves
  - onnx
library_name: onnx
---

# Astromer 2

> Part of the **[light-curve](https://github.com/light-curve)** family of open-source tools for astronomical time-series analysis.
> Available from Python via the [`light-curve`](https://light-curve.snad.space/) package — `pip install light-curve`.
> Documentation: <https://light-curve.snad.space/>

**HuggingFace:** [light-curve/astromer2](https://huggingface.co/light-curve/astromer2)

## Paper

Donoso-Oliva, C., Becker, I., Protopapas, P., Cabrera-Vives, G., Cádiz-Leyton, M., & Moreno-Cartagena, D. (2026). *Generalizing across astronomical surveys: Few-shot light curve classification with Astromer 2*. Astronomy & Astrophysics (in press).

```bibtex
@article{astromer2,
  author  = {Donoso-Oliva, C. and Becker, I. and Protopapas, P. and
             Cabrera-Vives, G. and C{\'a}diz-Leyton, M. and Moreno-Cartagena, D.},
  title   = {Generalizing across astronomical surveys: Few-shot light curve
             classification with {Astromer} 2},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  note    = {In press},
}
```

## Original code

<https://github.com/astromer-science/main-code> (git submodule at `models/astromer2/code/`)

## License

MIT — see [LICENSE](LICENSE).

## Model overview

Astromer 2 is a BERT-inspired transformer encoder pretrained on 1.5 million MACHO light curves via masked magnitude prediction. The encoder processes irregularly-sampled photometric time series (time, magnitude) using MJD-aware positional encoding and a trainable mask token. It produces per-timestep contextual embeddings that can be aggregated into a fixed-size representation for downstream tasks such as few-shot classification.

Default configuration: 6 attention blocks, 4 heads, head dimension 64 (d_model = 256), sequence length 200, embedding dimension 256.

## Input data format

Raw light curves are pairs `(time, mag)`:
- `time` — observation time in days. Need not be absolute MJD; any consistent time axis in days works because the pipeline subtracts the per-window mean before the encoder sees it. The pretrained weights were produced from MACHO data with MJD ~48800–51700.
- `mag` — magnitude. MACHO instrumental magnitudes are typically negative (e.g. −10 to −3); the pipeline is not restricted to that range.

Photometric errors are **not used** at inference. The upstream preprocessing code expects a 3-column `[time, mag, err]` array internally, but errors only appear in the pretraining reconstruction-loss weights (`outputs['w_error']`), which are never passed to the encoder. Pass dummy zeros if you run the pipeline directly.

## Preprocessing steps

All steps are implemented in `code/src/data/loaders.py` (`get_loader`) and `code/src/data/preprocessing.py`.

### Step 1 — Windowing

The upstream code supports two windowing strategies via the `sampling` flag of `to_windows`:

- **`sampling=True` — random window** (used during pretraining): a single contiguous window of 200 observations is drawn at a uniformly random starting position. Light curves shorter than 200 observations are used in full.
- **`sampling=False` — sequential windows** (used for test-data generation): the light curve is divided into sequential, non-overlapping windows of 200 observations. A light curve of length *L* yields ⌊*L*/200⌋ + 1 windows; the last window may be shorter than 200 and is padded in step 3. Light curves shorter than 200 observations produce a single window. When a light curve produces multiple windows, each window yields a separate embedding vector; to obtain a single per-light-curve embedding, average the per-window embeddings.

Test-data is generated with `sampling=False`.

Source: `src/data/preprocessing.py:to_windows`.

### Step 2 — Zero-mean normalization

Subtract the per-window column mean from each column:

```
x_norm = x - mean(x, axis=0)   # x has shape [n_obs, 3]; columns: time, mag, err
```

After this step `times` = time − mean(time) and `input` = mag − mean(mag) are centred around zero.

Source: `src/data/preprocessing.py:standardize`.

### Step 3 — Padding and mask construction

Right-pad the normalised sequence to exactly 200 time steps with zeros. Construct `mask_in`:

```
mask_in[i] = 0   for i < n_obs   (real observation — visible to encoder)
mask_in[i] = 1   for i >= n_obs  (padding — hidden from encoder)
```

> **Note on mask convention:** the internal pipeline uses `mask_in=0` for visible positions and `mask_in=1` for padding/hidden positions. This is the opposite of the ONNX interface (see below).

Source: `src/data/masking.py:mask_sample`, padding block at the end.

### Step 4 — Format encoder inputs

Extract the two encoder inputs from the normalised, padded array:

| Tensor | Source | Shape |
|--------|--------|-------|
| `input` | normalised magnitude column | `[batch, 200, 1]` |
| `times` | normalised time column | `[batch, 200, 1]` |
| `mask_in` | constructed in step 3 | `[batch, 200, 1]` |

The normalised error column is **not** fed to the encoder. Errors appear only in the pretraining reconstruction loss.

Source: `src/data/loaders.py:format_inp_astromer` (`aversion='base'`).

## Inputs (ONNX)

The exported ONNX models use a **user-friendly mask convention** that is the inverse of the internal pipeline:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input` | `[batch, 200, 1]` | `mag − mean(mag)` over the window (step 2 above) |
| `times` | `[batch, 200, 1]` | `time − mean(time)` over the window (step 2 above) |
| `mask_in` | `[batch, 200, 1]` | **1 = valid observation, 0 = padding** |

The ONNX wrapper inverts `mask_in` internally before passing it to the encoder, so consumers can use the intuitive convention.

## Outputs (ONNX)

Single file `astromer2.onnx` with three named outputs:

| Output name | Shape | Aggregation |
|-------------|-------|-------------|
| `mean` | `[batch, 256]` | Masked mean pooling: `sum(z * mask_in) / sum(mask_in)` |
| `max`  | `[batch, 256]` | Masked max pooling over valid timesteps |
| `sequence` | `[batch, 200, 256]` | Per-timestep features |

Request only the output(s) you need via `session.run(["mean"], feed)` — onnxruntime will prune unused computation.

ONNX opset: 13.

## Weights

Source: [Zenodo record 18207945](https://zenodo.org/records/18207945)
Training dataset: MACHO (1.5 million light curves, V and R bands)
Checkpoint: `astromer_v2/macho/`

The test-data parquet file was generated with these MACHO weights and `sampling=False` (sequential windows).
