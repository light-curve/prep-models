---
tags:
  - astronomy
  - time-series
  - light-curves
  - onnx
library_name: onnx
---

# ATCAT

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

| File | Shape | Aggregation |
|------|-------|-------------|
| `atcat_token.onnx` | `[batch, 384]` | Hidden state used by the upstream classifier (last valid token) |
| `atcat_mean.onnx` | `[batch, 384]` | Masked mean pool of transformer outputs |
| `atcat_full.onnx` | `[batch, 243, 384]` | Full padded transformer output sequence |

## Preprocessing steps

1. Use the upstream ATCAT ELAsTiCC-derived Parquet data format or convert your data into the same padded-per-object sequence fields.
2. Keep sequence order chronological as expected by the upstream preprocessing.
3. Pad sequences to length 243 and set `mask=0` for padding positions.
4. Encode LSST bands as `u, g, r, i, z, Y -> 0, 1, 2, 3, 4, 5`.

## Weights

Source: Google Drive archive linked from the upstream ATCAT README (`atcat_derived_data.tar`)

Model path used by this wrapper:
`results/elasticc/CORE/lc_only_cv_0/checkpoints/model_40000.pt`

Dataset used by this wrapper:
`data_parquet/split_0/test_*.parquet`
