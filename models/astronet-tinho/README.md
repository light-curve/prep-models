# AstroNet-TINHO

## Paper

Tarek Allam Jr., Tarek Allam Jr.
"Paying Attention to Astronomical Transients: Photometric Classification with the Time-Series Transformer."
*arXiv*, 2021.

```bibtex
@article{allam2021t2,
  title   = {Paying Attention to Astronomical Transients:
             Photometric Classification with the Time-Series Transformer},
  author  = {Allam, Tarek Jr.},
  journal = {arXiv preprint arXiv:2105.06178},
  year    = {2021}
}
```

TINHO (the compressed variant) is described in:

Tarek Allam Jr.
"Optimised and Compressed Attention for Light Curves."
*arXiv*, 2023.

```bibtex
@article{allam2023tinho,
  title   = {Optimised and Compressed Attention for Light Curves},
  author  = {Allam, Tarek Jr.},
  journal = {arXiv preprint arXiv:2303.08951},
  year    = {2023}
}
```

## Original code

https://github.com/tallamjr/astronet (git submodule at `models/astronet-tinho/code/`)

## License

Apache-2.0 — see `LICENSE`.

## Model overview

TINHO is a compressed version of the T2 time-series transformer, designed to classify
14 types of astronomical transients from PLAsTiCC photometric light curves.  The
architecture uses a convolutional embedding layer, sinusoidal positional encoding, a
single self-attention TransformerBlock, and global average pooling, followed by a
weight-clustered Dense softmax classifier (16 weight centroids via
`tensorflow_model_optimization`).  This clustering reduces model size by ~18× relative
to the original T2 while preserving classification performance.

The model was deployed on real Zwicky Transient Facility data via the FINK alert
broker.

Two trained variants are available (both included in the upstream repository):

| Variant   | Bands      | Redshift | Sequence length |
|-----------|------------|----------|-----------------|
| GR        | g, r       | No       | 100             |
| UGRIZY    | u,g,r,i,z,y | Yes (host galaxy) | 100       |

## Inputs

### `tinho_gr.onnx`

| Tensor | Shape      | Description |
|--------|------------|-------------|
| flux   | [B, 100, 2] | GP-interpolated flux grid for g and r bands |

### `tinho_ugrizy.onnx`

| Tensor   | Shape      | Description |
|----------|------------|-------------|
| flux     | [B, 100, 6] | GP-interpolated flux grid for all 6 LSST bands (u,g,r,i,z,y) |
| redshift | [B, 2]     | Host-galaxy features: [hostgal_photoz, hostgal_photoz_err] |

## Outputs (ONNX)

| File                  | Output shape | Description |
|-----------------------|--------------|-------------|
| `tinho_gr.onnx`       | [B, 14]      | Softmax class probabilities (14 PLAsTiCC classes) |
| `tinho_ugrizy.onnx`   | [B, 14]      | Softmax class probabilities (14 PLAsTiCC classes) |

Output index → PLAsTiCC class ID → class name:

| Index | Class ID | Name        |
|-------|----------|-------------|
| 0     | 90       | SNIa        |
| 1     | 67       | SNIa-91bg   |
| 2     | 52       | SNIax       |
| 3     | 42       | SNII        |
| 4     | 62       | SNIbc       |
| 5     | 95       | SLSN-I      |
| 6     | 15       | TDE         |
| 7     | 64       | KN          |
| 8     | 88       | AGN         |
| 9     | 92       | RRL         |
| 10    | 65       | M-dwarf     |
| 11    | 16       | EB          |
| 12    | 53       | Mira        |
| 13    | 6        | μ-Lens-Single |

**Note:** the class-index ordering above is the one produced by the label encoder
used during training (`sklearn.preprocessing.LabelEncoder` fit on sorted class IDs).
Verify with the `get_encoding` utility in the upstream repo if in doubt.

## Preprocessing steps

The upstream code uses Gaussian Process (GP) interpolation to convert sparse,
irregularly-sampled photometry to a regular 100-point time grid:

1. **Trim** the light curve to the transient window (±50 days around first and last
   detected point).
2. **GP interpolation** using a Matérn kernel in wavelength × time space
   (see `astronet/preprocess.py → generate_gp_all_objects`).  This produces a
   regular grid of 100 timestamps spanning the trimmed window.
3. **Per-band flux grid**: shape `(100, n_bands)`.
4. **Band order**: g, r for GR model; u, g, r, i, z, y (LSST filter order) for
   UGRIZY model.
5. **Redshift** (UGRIZY only): `[hostgal_photoz, hostgal_photoz_err]` from the
   PLAsTiCC object table, shape `(2,)` per object.
6. **No explicit normalization** is applied beyond what the GP fit produces; the
   model is not scale-invariant so fluxes should be in the same units as the
   PLAsTiCC training data (µJy).

## Weights

Source: included in the upstream git repository at
`code/astronet/tinho/models/plasticc/`.  No separate download is needed; the
submodule checkout includes the SavedModel files.

Dataset: PLAsTiCC simulated survey data (LSST), 14-class classification,
Gaussian Process pre-processed to a 100-step regular grid.

## Export

```bash
uv run --project models/astronet-tinho python -m astronet_tinho_prep export \
    --out-dir models/astronet-tinho/out/onnx
```
