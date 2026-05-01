# Astromer 2

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

The model was pretrained on MACHO survey photometry. MACHO light curves consist of triples `(mjd, mag, err)` where:
- `mjd` — Modified Julian Date of each observation (~48800–51700 for MACHO)
- `mag` — MACHO instrumental magnitude (typically negative values, e.g. −10 to −3 in the MACHO system)
- `err` — photometric error; some observations carry large negative sentinel values (e.g. −3000, −9000) indicating bad data — **these are passed through the pipeline as-is without filtering**

## Preprocessing pipeline

All steps are implemented in `code/src/data/loaders.py` (`get_loader`) and reproduced in this package's `utils/prep_models_utils/astromer/common.py` (`preprocess_curves`).

### Step 1 — Windowing

If the light curve has more than 200 observations, take the first 200 (non-random, sequential window). If it has fewer than 200, use all observations and pad in step 3.

Source: `src/data/preprocessing.py:to_windows` with `sampling=False`.

### Step 2 — Zero-mean normalization

Subtract the per-light-curve column mean from **all three columns** (time, magnitude, error):

```
x_norm = x - mean(x, axis=0)   # x has shape [n_obs, 3]
```

After this step, `times` and `input` (magnitudes) are centred around zero. The error column is also normalised but is discarded before the encoder (see step 4).

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

## ONNX interface

The exported ONNX models use a **user-friendly mask convention** that is the inverse of the internal pipeline:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input` | `[batch, 200, 1]` | Zero-mean normalised magnitudes (step 2 above) |
| `times` | `[batch, 200, 1]` | Zero-mean normalised times (step 2 above) |
| `mask_in` | `[batch, 200, 1]` | **1 = valid observation, 0 = padding** |

The ONNX wrapper inverts `mask_in` internally before passing it to the encoder, so consumers can use the intuitive convention.

## Outputs (ONNX)

| File | Output shape | Aggregation |
|------|-------------|-------------|
| `astromer2_mean.onnx` | `[batch, 256]` | Masked mean pooling: `sum(z * mask_in) / sum(mask_in)` |
| `astromer2_max.onnx`  | `[batch, 256]` | Masked max pooling over valid timesteps |
| `astromer2_full.onnx` | `[batch, 200, 256]` | Full per-timestep sequence; consumer aggregates |

## Weights

Source: Zenodo record [10.5281/zenodo.18207945](https://zenodo.org/doi/10.5281/zenodo.18207945)
Training dataset: MACHO (1.5 million light curves, V and R bands)
Checkpoint: `astromer_v2/macho/`

## ONNX on HuggingFace

Published at: <https://huggingface.co/light-curve/astromer2>
License: MIT (same as original model)
