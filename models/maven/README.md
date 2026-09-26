---
tags:
  - astronomy
  - time-series
  - light-curves
  - supernovae
  - onnx
library_name: onnx
license: mit
---

# Maven (light-curve encoder)

**Part of the [light-curve](https://github.com/light-curve) family of open-source tools for astronomical time-series analysis.**

Available from Python via the [`light-curve`](https://light-curve.snad.space/) package: `pip install light-curve`. Documentation: [light-curve.snad.space](https://light-curve.snad.space/).

**HuggingFace:** [light-curve/maven](https://huggingface.co/light-curve/maven)

## Paper

Zhang, G., Helfer, T., Gagliano, A. T., Mishra-Sharma, S., Villar, V. A. (2024). *Maven: A Multimodal Foundation Model for Supernova Science*. [arXiv:2408.16829](https://arxiv.org/abs/2408.16829).

```bibtex
@article{zhang2024maven,
  author = {Zhang, Gemma and Helfer, Thomas and Gagliano, Alexander T. and Mishra-Sharma, Siddharth and Villar, V. Ashley},
  title = {Maven: A Multimodal Foundation Model for Supernova Science},
  journal = {arXiv preprint arXiv:2408.16829},
  year = {2024},
  doi = {10.48550/arXiv.2408.16829}
}
```

## Original code

<https://github.com/ThomasHelfer/multimodal-supernovae> (git submodule at `models/maven/code/`)

## License

MIT — see [LICENSE](LICENSE). The checkpoints are distributed in the same repository under the same license.

## Model overview

Maven aligns supernova photometry and spectroscopy in a shared latent space with a CLIP-style contrastive objective.
Each modality has its own transformer encoder; the light-curve encoder is a 5-block transformer (embedding dim 64, 8 heads) with sinusoidal time encoding, a linear magnitude embedding and a learned per-band embedding for ZTF R and g.
Maven is first pre-trained on simulated (noiseless) ZTF light curves with matching spectra, then fine-tuned on real ZTF Bright Transient Survey (BTS) light curves paired with classification spectra.

This integration exports the **light-curve encoder only** of the fine-tuned Maven model.
Upstream fine-tunes one model per stratified 5-fold split; we export fold 0 (`gallant-sweep-1`), using the smallest-epoch checkpoint as upstream `evaluate_models.py` does.

## Inputs

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `mag` | `[batch, 200]` | float32 | Apparent magnitude; R band in positions 0–99, g band in 100–199 |
| `time` | `[batch, 200]` | float32 | Time in days since the first observation **of the same band** |
| `mask` | `[batch, 200]` | float32 | `1` for valid observations, `0` for padding |

The sequence length is fixed to the training layout: 100 slots per band, R first, then g.

## Outputs (ONNX)

Single file `maven.onnx` with three named outputs:

| Output | Shape | Aggregation |
|--------|-------|-------------|
| `mean` | `[batch, 64]` | Masked mean pool of transformer outputs |
| `clip` | `[batch, 128]` | Maven CLIP embedding: `mean` → encoder projection (64→32) → CLIP projection (32→128) → L2 normalization |
| `sequence` | `[batch, 200, 64]` | Full per-timestep transformer outputs (unmasked) |

`clip` is the representation used in the paper (shared light-curve/spectrum space).

## Preprocessing steps

Following upstream `src/dataloader.py::load_lightcurves`:

1. Correct magnitudes for Milky Way extinction with the Cardelli, Clayton & Mathis (1989) law, R_V = 3.1 (upstream uses `extinction.ccm89` with its own effective-wavelength values: g → 1196.25 Å, R → 6366.38 Å).
2. Split the light curve by band into R and g.
3. For each band: if it has more than 100 observations, randomly sample 100 without replacement; observation order does not matter.
4. For each band: subtract the time of the band's earliest kept observation.
5. Zero-pad each band to 100 slots and build the mask (`1` = observed).
6. Concatenate R then g into 200-element `mag`, `time` and `mask` arrays.

Magnitudes are used as-is (no normalization). Upstream fine-tuning adds Gaussian noise scaled by `magerr` as augmentation; at inference `magerr` is not used.

## Test data

`out/test-data/maven_test.parquet` contains synthetic ZTF-like two-band supernova light curves (Bazin flux model, irregular cadence, 5–150 observations per band), the preprocessed `input_mag`/`input_time`/`input_mask` arrays and the `embedding_mean`/`embedding_clip` outputs.
Real ZTF BTS data is only available via the upstream [HuggingFace dataset](https://huggingface.co/datasets/thelfer/multimodal_supernovae); extinction correction is skipped for synthetic data.

## Weights

Source: <https://github.com/ThomasHelfer/multimodal-supernovae/tree/1f571aa9311eccce6358bfa1ff0fa5e8212d1422/models/clip_noiselesssimpretrain_clipreal/gallant-sweep-1> (`epoch=25-step=3042.ckpt`)

Dataset: simulated ZTF light curves + spectra (pre-training), ZTF Bright Transient Survey light curves + spectra (fine-tuning); see <https://huggingface.co/datasets/thelfer/multimodal_supernovae>.
