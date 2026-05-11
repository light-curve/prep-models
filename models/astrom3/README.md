---
tags:
  - astronomy
  - time-series
  - light-curves
  - variable-stars
  - onnx
library_name: onnx
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

MIT License — see [LICENSE](LICENSE).

## Model overview

AstroM3 is a self-supervised multimodal contrastive model for variable-star classification that jointly trains photometry (light-curve), spectra, and metadata encoders using a CLIP-style objective. This integration exports the **photo-only encoder** from the pretrained CLIP checkpoint (`AstroMLCore/AstroM3-CLIP-photo`) as an ONNX embedding model.

The photo encoder is an [Informer](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) transformer (ProbSparse attention, 8 layers, d_model=128) trained on ZTF variable-star light curves from the MACC dataset. For ONNX export, the ProbSparse attention layers are replaced with standard scaled dot-product attention, which is equivalent in expectation and fully ONNX-exportable.

## Inputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `x_enc` | `[batch, 200, 9]` | Padded photometry features (9 channels per timestep — see preprocessing) |
| `mask` | `[batch, 200]` | `1` for valid timesteps, `0` for padding |

## Outputs (ONNX)

Single file `astrom3.onnx` with three named outputs:

| Output | Shape | Aggregation |
|--------|-------|-------------|
| `mean` | `[batch, 128]` | Masked mean pool of encoder outputs |
| `max` | `[batch, 128]` | Masked max pool of encoder outputs |
| `sequence` | `[batch, 200, 128]` | Full per-timestep encoder outputs (unmasked) |

## Preprocessing steps

The 9 input channels per timestep are, in order:

| Index | Feature |
|-------|---------|
| 0 | Time (MJD) |
| 1 | Magnitude / flux |
| 2 | Magnitude / flux error |
| 3 | Amplitude |
| 4 | Period (Lomb-Scargle) |
| 5 | LKSL statistic |
| 6 | RFR score |
| 7 | MAD (median absolute deviation) |
| 8 | Δt (time between consecutive observations) |

Global features (amplitude, period, LKSL, RFR, MAD) are replicated across all timesteps following the upstream preprocessing in `AstroMLCore/AstroM3Processed`.

1. Build the per-timestep feature matrix as described above.
2. Sort observations chronologically.
3. Pad or center-crop to 200 timesteps; set `mask=0` for padded positions.
4. Use `float32` for all tensors.

## Weights

Source: <https://huggingface.co/AstroMLCore/AstroM3-CLIP-photo>

The `model.safetensors` file contains the full CLIP model; this wrapper extracts weights under the `photometry_encoder.*` prefix and loads them into a standalone Informer.

Dataset: ZTF variable-star light curves from the MACC catalog (`AstroMLCore/AstroM3Processed`).
