---
tags:
  - astronomy
  - time-series
  - light-curves
  - variable-stars
  - onnx
library_name: onnx
license: cc-by-4.0
---

# AstroM3 (photo encoder)

## Paper

Rizhko, M. et al. (2024). *AstroM³: A self-supervised multimodal model for astronomy*. arXiv:2411.08842.

```bibtex
@article{rizhko2024astrom3,
  author = {Rizhko, Mariia and Bloom, Joshua S.},
  title = {{AstroM³}: A self-supervised multimodal model for astronomy},
  journal = {arXiv preprint arXiv:2411.08842},
  year = {2024}
}
```

## Original code

<https://github.com/MeriDK/AstroM3> (git submodule at `models/astrom3/code/`)

## License

- **Code** (this repository): MIT — see [LICENSE](LICENSE).
- **Model weights** (`AstroMLCore/AstroM3-CLIP-photo`): Creative Commons Attribution 4.0 (CC BY 4.0).

## Model overview

AstroM3 is a self-supervised multimodal contrastive model for variable-star classification that jointly trains photometry (light-curve), spectra, and metadata encoders using a CLIP-style objective. This integration exports the **photo-only encoder** from the pretrained CLIP checkpoint (`AstroMLCore/AstroM3-CLIP-photo`) as an ONNX embedding model.

The photo encoder is an [Informer](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) transformer (ProbSparse attention, 8 layers, d_model=128) trained on ZTF variable-star light curves from the MACC dataset. For ONNX export, the ProbSparse attention layers are replaced with standard scaled dot-product attention, which is equivalent in expectation and fully ONNX-exportable.

## Inputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `x_enc` | `[batch, 200, 9]` | Padded photometry features (9 channels per timestep — see preprocessing) |
| `mask` | `[batch, 200]` | `1` for valid timesteps, `0` for padding |

## Outputs (ONNX)

Single file `astrom3.onnx` with two named outputs:

| Output | Shape | Aggregation |
|--------|-------|-------------|
| `mean` | `[batch, 128]` | Masked mean pool of encoder outputs |
| `sequence` | `[batch, 200, 128]` | Full per-timestep encoder outputs (unmasked) |

## Preprocessing steps

The 9 input channels per timestep are built by `preprocess_lc()` in the
upstream dataset (`AstroMLCore/AstroM3Dataset`):

| Index | Feature | How obtained |
|-------|---------|--------------|
| 0 | `time` (HJD scaled to [0, 1]) | per-observation |
| 1 | `flux` = `(flux − mean) / MAD` | per-observation |
| 2 | `flux_err` = `flux_err / MAD` | per-observation |
| 3 | `amplitude` | **ASAS-SN catalog scalar, replicated to every timestep** |
| 4 | `period` | **ASAS-SN catalog scalar, replicated** |
| 5 | `lksl_statistic` (Lafler-Kinman string length) | **ASAS-SN catalog scalar, replicated** |
| 6 | `rfr_score` (Random Forest classifier score) | **ASAS-SN catalog scalar, replicated** |
| 7 | `log10(MAD_flux)` | global scalar computed from LC, replicated |
| 8 | `delta_t` = `(max_HJD − min_HJD) / 365` | global scalar computed from LC, replicated |

Features 3–6 come directly from the ASAS-SN v-band variable-star catalog
(Jayasinghe et al. 2019) and are **not recomputed** from the light curve by
this codebase. Users applying this model to non-ASAS-SN data must provide
equivalent values (e.g. run a Lomb-Scargle period finder and compute
peak-to-peak amplitude themselves).

Preprocessing recipe for a single light curve:

1. Deduplicate and sort observations by HJD.
2. Compute `mean` and `MAD` of the flux column; normalize flux and flux_err.
3. Scale HJD to [0, 1] over the span of the light curve.
4. Compute `log10(MAD_flux)` and `delta_t = (max_HJD − min_HJD) / 365`.
5. Obtain `amplitude`, `period`, `lksl_statistic`, `rfr_score` from the
   ASAS-SN catalog (or compute equivalents).
6. Tile the 6 global scalars across all timesteps; concatenate with columns
   0–2 to produce an `(N, 9)` array.
7. Pad or center-crop to 200 timesteps; set `mask = 0` for padded positions.
8. Use `float32` for all tensors.

## Weights

Source: <https://huggingface.co/AstroMLCore/AstroM3-CLIP-photo>

The `model.safetensors` file contains the full CLIP model; this wrapper extracts weights under the `photometry_encoder.*` prefix and loads them into a standalone Informer.

Dataset: ZTF variable-star light curves from the MACC catalog (`AstroMLCore/AstroM3Processed`).
