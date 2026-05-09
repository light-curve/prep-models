---
tags:
  - astronomy
  - time-series
  - light-curves
  - onnx
library_name: onnx
---

# ATCAT

**HuggingFace:** [light-curve/atcat](https://huggingface.co/light-curve/atcat)

## Paper

Tung, Z. (2025). *ATCAT: Astronomical Timeseries CAusal Transformer*. arXiv:2511.00614.

```bibtex
@article{tung2025atcat,
  author = {Tung, Zora},
  title = {{ATCAT}: Astronomical Timeseries CAusal Transformer},
  journal = {arXiv preprint arXiv:2511.00614},
  year = {2025}
}
```

## Original code

<https://codeberg.org/zorat/atcat> (git submodule at `models/atcat/code/`)

## License

ATCAT is distributed upstream under a modified MIT license with a non-military-use restriction.
See [LICENSE](LICENSE) and the upstream `README.md` for the exact terms.

## Model overview

This integration exports the upstream ATCAT light-curve-only ELAsTiCC classifier as an ONNX embedding model. ATCAT is a causal transformer for irregularly sampled astronomical time series. The exported wrapper uses the real upstream light-curve embedder and transformer stack from the `lc_only(split=0)` checkpoint, and exposes hidden representations before the final classifier head.

The current export targets the upstream LC-only core model (`results/elasticc/CORE/lc_only_cv_0`). The LC+metadata variant is intentionally not wrapped yet because the upstream README notes that the saved metadata preprocessing artifacts are incomplete for out-of-the-box reuse.

## Inputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `flux` | `[batch, 243]` | Padded calibrated flux values |
| `flux_err` | `[batch, 243]` | Padded flux uncertainties |
| `time` | `[batch, 243]` | Padded observation times |
| `mask` | `[batch, 243]` | `1` for valid points, `0` for padding |
| `channel_index` | `[batch, 243]` | LSST band indices in ATCAT order: `u=0, g=1, r=2, i=3, z=4, Y=5` |

## Outputs (ONNX)

Two files are produced with the same three named outputs:

Two files are produced, both with the same three named outputs:

| Output name | Shape | Aggregation |
|-------------|-------|-------------|
| `last` | `[batch, 384]` | Hidden state at the last valid LC observation (position `num_lc_points-1`) |
| `mean`  | `[batch, 384]` | Masked mean pool of transformer outputs |
| `sequence` | `[batch, 243, 384]` | Per-timestep transformer features (`last` is the final valid element of this) |

`atcat_bf16.onnx` is the direct export (bfloat16 weights). `atcat_f32.onnx` is generated automatically by `prep-models atcat export` by stripping the bfloat16 casts.

Request only the output(s) you need via `session.run(["token"], feed)` — onnxruntime will prune unused computation.

## Preprocessing steps

Prepare inputs using the same pipeline applied during training (`step1_mask_alert_and_join.py`, `polars_ds.py`):

1. **Merge all bands into a single chronological sequence.** Collect all observations across all six bands, sort by time, and represent each observation's band as a `channel_index`: `u=0, g=1, r=2, i=3, z=4, Y=5`.

2. **Shift time to start at zero.** Subtract the minimum MJD in the object's light curve from all observation times. Supply time in **days**.

3. **No flux normalisation.** Pass raw flux and flux_err values without any normalisation. Both are in SNANA FLUXCAL units with reference zero point ZP = 27.5 (a source at 27.5 AB mag has FLUXCAL = 1). The model applies a multi-scale tanh transformation internally.

4. **Zero-pad to length 243.** If the object has fewer than 243 observations, right-pad `flux`, `flux_err`, `time`, and `channel_index` with zeros to length 243.

5. **Set the mask.** Set `mask = 1` for every slot containing a real observation and `mask = 0` for every padding slot.

> **Note:** The training data was filtered to the transient detection window: observations from 30 days before the first detection through the last detected photometry. Data far outside this window is out of distribution for this model.

## Weights

Source: Google Drive archive linked from the upstream ATCAT README (`atcat_derived_data.tar`)

Model path used by this wrapper:
`results/elasticc/CORE/lc_only_cv_0/checkpoints/model_40000.pt`

Dataset used by this wrapper:
`data_parquet/split_0/test_*.parquet`
