---
license: gpl-3.0
tags:
  - astronomy
  - time-series
  - light-curves
  - onnx
library_name: onnx
---

# Astromer 1 (ZTF DR20 g-band)

> Part of the **[light-curve](https://github.com/light-curve)** family of open-source tools for astronomical time-series analysis.
> Available from Python via the [`light-curve`](https://light-curve.snad.space/) package — `pip install light-curve`.
> Documentation: <https://light-curve.snad.space/>

**HuggingFace:** [light-curve/astromer1-ztfdr20](https://huggingface.co/light-curve/astromer1-ztfdr20)

## Paper

Nakoneczny, S. J., Bilicki, M., Pollo, A., Hui, A. Y. W., Bianco, M., Lares-Martiz, M., & Marchetti, L. (2025). *QZO: A Catalog of 5 Million Quasars from the Zwicky Transient Facility*. The Astrophysical Journal, 992, 153.

```bibtex
@article{nakoneczny2025qzo,
  author  = {Nakoneczny, S.~J. and Bilicki, M. and Pollo, A. and
             Hui, A.~Y.~W. and Bianco, M. and Lares-Martiz, M. and
             Marchetti, L.},
  title   = {{QZO}: A Catalog of 5 Million Quasars from the Zwicky
             Transient Facility},
  journal = {The Astrophysical Journal},
  volume  = {992},
  pages   = {153},
  year    = {2025},
  doi     = {10.3847/1538-4357/adcbf0},
}
```

Original Astromer 1 architecture:

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

<https://github.com/snakoneczny/ztf-agn> (training scripts)

Encoder architecture: <https://github.com/astromer-science/main-code> (v1.0 tag, git submodule at `models/astromer1-ztfdr20/code/`)

## License

GPL-3.0 — see [LICENSE](LICENSE).

## Model overview

This is the Astromer 1 transformer encoder retrained on ZTF DR20 g-band light curves
by Nakoneczny et al. (2025) for quasar/galaxy/star classification.  The architecture
is identical to the original MACHO-trained Astromer 1: 2 transformer layers, 4
attention heads, 256-dimensional embeddings.

The encoder weights are extracted from the `ANN_clf` checkpoint (full FCATT model:
encoder + FC classification head) published on Zenodo.  Only the encoder sublayer is
exported to ONNX; the classification head is discarded.

## Inputs

All tensors are `float32`. Both magnitudes and times are **zero-mean normalized** before
passing to the model (subtract the per-window mean of each).

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input` | `[batch, 200, 1]` | `mag − mean(mag)` over the window |
| `times` | `[batch, 200, 1]` | `time − mean(time)` over the window |
| `mask_in` | `[batch, 200, 1]` | 1 = valid observation, 0 = padded position |

## Outputs (ONNX)

Single file `astromer1_ztfdr20.onnx` with three named outputs:

| Output name | Shape | Aggregation |
|-------------|-------|-------------|
| `mean` | `[batch, 256]` | Masked mean pooling over valid timesteps |
| `max`  | `[batch, 256]` | Masked max pooling over valid timesteps |
| `sequence` | `[batch, 200, 256]` | Per-timestep features |

Request only the output(s) you need via `session.run(["mean"], feed)` — onnxruntime will prune unused computation.

ONNX opset: 13.

## Preprocessing steps

Photometric errors are **not used** at inference.

1. **Collect** ZTF g-band observation times (in days) and magnitudes.
2. **Truncate** each light curve to at most 200 observations.
3. **Zero-mean normalize** both columns over the window:
   `time -= time.mean()`, `mag -= mag.mean()`
4. **Pad** shorter light curves to exactly 200 positions: append zeros to both `input` and `times`.
5. **Build the mask**: set `mask_in = 1` for real observations, `mask_in = 0` for padded positions.
6. **Reshape** each tensor to `[batch, 200, 1]` (add trailing dimension).

The sequence length is fixed at 200 by the pretrained weights.

## Weights

Source: [Zenodo record 16410988](https://zenodo.org/records/16410988) (`ANN_clf.*` files)
Training dataset: ZTF DR20 g-band light curves cross-matched with SDSS
