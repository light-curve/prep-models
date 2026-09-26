# AstraCLR

**Part of the [light-curve](https://github.com/light-curve) family of open-source tools for astronomical time-series analysis.**

Available from Python via the [`light-curve`](https://light-curve.snad.space/) package: `pip install light-curve`. Documentation: [light-curve.snad.space](https://light-curve.snad.space/).

## Paper

Majumder, T., Malanchev, K., Ishida, E. E. O. (2026). *Multi-Scale Contrastive Attention for Light-Curve Representation Learning*. [arXiv:2606.31627](https://arxiv.org/abs/2606.31627).

```bibtex
@ARTICLE{2026arXiv260631627M,
       author = {{Majumder}, Torsha and {Malanchev}, Konstantin and {Ishida}, Emille E.~O.},
        title = "{Multi-Scale Contrastive Attention for Light-Curve Representation Learning}",
      journal = {arXiv e-prints},
     keywords = {Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = jun,
          eid = {arXiv:2606.31627},
        pages = {arXiv:2606.31627},
          doi = {10.48550/arXiv.2606.31627},
archivePrefix = {arXiv},
       eprint = {2606.31627},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260631627M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Original model

GitHub: https://github.com/TorshaMajumder/astra (private)

HuggingFace: https://huggingface.co/ashrot/astra-clr-base

Inference library: https://github.com/snad-space/astra-infer

## License

MIT

## Model overview

AstraCLR is a contrastive-learning encoder for multi-band photometric light curves.
It maps a ZTF (g, r, i) light curve to a 512-dimensional embedding via a transformer
architecture trained with a contrastive objective.  Unlike ASTROMER, the model accepts
multi-band input by concatenating per-band observation sequences with a log-wavelength
channel.  The ONNX file is distributed pre-built and requires no ML-framework conversion.

## Inputs

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `input` | `[batch, 700, 1]` | float32 | Inverse-variance-weighted mean-subtracted magnitude per band window |
| `times` | `[batch, 700, 1]` | float32 | Observation time minus MJD offset (58 000) |
| `band_info` | `[batch, 700, 1]` | float32 | lg(effective wavelength in Å) for the observation's band |
| `mask` | `[batch, 700]` | float32 | 0 = real observation, 1 = padded |

The 700-element sequence is a concatenation of three per-band windows:
g (0–299, 300 obs), r (300–649, 350 obs), i (650–699, 50 obs).

## Output

| Tensor | Shape | Description |
|--------|-------|-------------|
| mean | `[batch, 512]` | Light-curve embedding |

## Preprocessing steps

1. Sort observations chronologically within each band.
2. Select the first `SEQ_PER_BAND[band]` observations ("beginning" strategy).
3. **Magnitude normalisation** per band: subtract the inverse-variance-weighted mean.
   `norm_mag = mag − Σ(mag/magerr²) / Σ(1/magerr²)`
4. **Time normalisation**: `norm_time = mjd − 58 000`
5. **Band channel**: `band_info = lg(eff_wavelength_Å)`
   — g: lg(4746.48), r: lg(6366.38), i: lg(7829.03)
6. **Padding**: zero-pad shorter windows to the required length; set `mask = 1` for
   padded positions, `mask = 0` for real observations.

## Model file

HuggingFace (ONNX): https://huggingface.co/light-curve/astra-clr

Source: https://huggingface.co/ashrot/astra-clr-base/blob/main/astra-clr.onnx
Training data: ZTF Zubercal DR16 (https://huggingface.co/datasets/snad-space/astra-zubercaldr16_gaiadr3vclassre)
