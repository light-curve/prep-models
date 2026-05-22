# AstraCLR

## Paper

Majumder et al., 2026, in prep.

## Original model

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
| `band_info` | `[batch, 700, 1]` | float32 | log₁₀(effective wavelength in Å) for the observation's band |
| `mask` | `[batch, 700]` | float32 | 0 = real observation, 1 = padded |

The 700-element sequence is a concatenation of three per-band windows:
g (0–299, 300 obs), r (300–649, 350 obs), i (650–699, 50 obs).

## Output

| Tensor | Shape | Description |
|--------|-------|-------------|
| output | `[batch, 512]` | Light-curve embedding |

## Preprocessing steps

1. Sort observations chronologically within each band.
2. Select the first `SEQ_PER_BAND[band]` observations ("beginning" strategy).
3. **Magnitude normalisation** per band: subtract the inverse-variance-weighted mean.
   `norm_mag = mag − Σ(mag/magerr²) / Σ(1/magerr²)`
4. **Time normalisation**: `norm_time = mjd − 58 000`
5. **Band channel**: `band_info = log₁₀(eff_wavelength_Å)`
   — g: log₁₀(4746.48), r: log₁₀(6366.38), i: log₁₀(7829.03)
6. **Padding**: zero-pad shorter windows to the required length; set `mask = 1` for
   padded positions, `mask = 0` for real observations.

## Model file

Source: https://huggingface.co/ashrot/astra-clr-base/blob/main/astra-clr.onnx
Training data: ZTF photometric survey (g, r, i bands)
