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

<https://github.com/astromer-science/main-code> (git submodule at `models/astromer1/code/`, pinned to Astromer 1 commit)

## License

MIT — see [LICENSE](LICENSE).

## Model overview

Astromer 1 is a transformer encoder pretrained on MACHO light curves via masked magnitude prediction. Compared to Astromer 2, it uses a simpler fixed zero mask token (non-trainable) and a shallower architecture. The encoder maps irregularly-sampled photometric time series to per-timestep contextual embeddings using an MJD-aware sinusoidal positional encoding.

## Inputs

All tensors are `float32`. Magnitudes must be **zero-mean normalized** before passing to the model.

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input` | `[batch, 200, 1]` | Zero-mean normalized magnitudes |
| `times` | `[batch, 200, 1]` | Observation times in MJD |
| `mask_in` | `[batch, 200, 1]` | 1 = valid observation, 0 = padded/masked position |

## Outputs (ONNX)

| File | Output shape | Aggregation |
|------|-------------|-------------|
| `astromer1_mean.onnx` | `[batch, 256]` | Masked mean pooling |
| `astromer1_max.onnx` | `[batch, 256]` | Masked max pooling |
| `astromer1_full.onnx` | `[batch, 200, 256]` | Full per-timestep sequence |

*Note: embedding dimension may differ from Astromer 2; verify after export.*

## Preprocessing steps

Same as Astromer 2 — see [Astromer 2 README](../astromer2/README.md#preprocessing-steps).

The original implementation is in `code/src/data/loaders.py` and `code/src/data/preprocessing.py`.

## Weights

Source: Zenodo (record ID to be confirmed — see `models/astromer1/code/README.md`)
Training dataset: MACHO

## ONNX on HuggingFace

Published at: <https://huggingface.co/light-curve/astromer1>
License: MIT (same as original model)
