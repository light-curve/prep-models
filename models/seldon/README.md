# SELDON

## Paper

Jiezhong Wu, Jack O'Brien, Jennifer Li, M. S. Krafczyk, Ved G. Shah, Amanda R. Wasserman,
Daniel W. Apley, Gautham Narayan, Noelle I. Samia.
"SELDON: Supernova Explosions Learned by Deep ODE Networks."
*Proceedings of the AAAI Conference on Artificial Intelligence*, 2026.

```bibtex
@inproceedings{2026seldonI,
  title     = {SELDON: Supernova Explosions Learned by Deep ODE Networks},
  author    = {Jiezhong Wu and Jack O'Brien and Jennifer Li and M. S. Krafczyk
               and Ved G. Shah and Amanda R. Wasserman and Daniel W. Apley
               and Gautham Narayan and Noelle I. Samia},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

## Original code

https://github.com/skai-institute/seldon (git submodule at `models/seldon/code/`)

## License

BSD 3-Clause — see `LICENSE`.

## Model overview

SELDON is a variational autoencoder for irregular, multi-band supernova light curves.
The encoder is a Neural ODE GRU: it processes observations in reverse chronological
order, integrating the hidden state between each pair of timestamps with an adaptive
ODE solver (Tsit5), then updating with a GRU cell at each valid observation.
After the sequence loop, the final hidden state is evolved over a fixed time grid
and aggregated with a DeepSet MLP to produce a 64-dimensional embedding.
The decoder (not exported) reconstructs flux using Gaussian basis functions.

Trained on SNIa light curves in up to 6 photometric bands.

## Inputs

| Tensor    | Shape      | Description |
|-----------|------------|-------------|
| flux      | [B, 1]     | Normalized flux value for one observation |
| band_idx  | [B]        | Integer band index (0–5) |
| time      | [B]        | Observation time (raw, in days) |
| mask      | [B]        | `True` = valid observation, `False` = padding |

Observations must be sorted in **descending** time order (newest first) before
feeding into the glue loop.

## Outputs (ONNX)

ONNX export is split into four files because the torchode adaptive ODE solver
is not ONNX-traceable.  The Python/C++ caller implements the outer loop using
these building blocks.

| File                  | Inputs → Shape                                            | Outputs → Shape   | Description |
|-----------------------|-----------------------------------------------------------|-------------------|-------------|
| `seldon_embed.onnx`   | flux(B,1), band_idx(B,), time(B,)                         | x_emb(B,H), t_sc(B,) | Embed one time-step: flux + band → hidden; also returns learner-scaled time |
| `seldon_ode_step.onnx`| h(B,H), t0(B,), t1(B,)                                   | h_next(B,H)       | Advance hidden state from t0 to t1 (single RK4 step, replaces adaptive Tsit5) |
| `seldon_gru_cell.onnx`| x_emb(B,H), h(B,H), mask(B,)                             | h_next(B,H)       | GRU update; passes h through unchanged when mask=False |
| `seldon_deepset.onnx` | zs_t(B,N,H+1)                                            | z(B,H)            | DeepSet aggregation; N = 50, H = 128 |

where H = 128 (hidden_dim), N = 50 (N_points).

**Note on RK4 vs adaptive Tsit5:** The trained model uses an adaptive-step Tsit5
solver internally (via `torchode`).  For ONNX export this is replaced with a
single fixed RK4 step per interval.  For well-resolved light curves (small Δt
relative to the ODE dynamics) the numerical difference is negligible.

## Inference glue loop

```python
import numpy as np
import onnxruntime as ort

embed   = ort.InferenceSession("seldon_embed.onnx")
ode     = ort.InferenceSession("seldon_ode_step.onnx")
gru     = ort.InferenceSession("seldon_gru_cell.onnx")
deepset = ort.InferenceSession("seldon_deepset.onnx")

DT        = 1e-4   # avoids t0==t1 singularity
H         = 128
N_POINTS  = 50
T_MAX     = 1.0    # encoder.t_max from config
T_SCALE   = ...    # softplus(encoder.time_scaling), save this scalar at export time

def encode(flux, band_idx, time, mask):
    # flux, time: (T,)  band_idx: (T,) int  mask: (T,) bool
    # All sorted descending by time.  B=1 for simplicity here.
    T = len(flux)
    B = 1

    # Embed every observation
    x_emb, t_sc = [], []
    for i in range(T):
        xe, ts = embed.run(None, {
            "flux":     flux[i:i+1, None].astype(np.float32),
            "band_idx": band_idx[i:i+1].astype(np.int64),
            "time":     time[i:i+1].astype(np.float32),
        })
        x_emb.append(xe)    # (1, H)
        t_sc.append(ts[0])  # scalar

    # RNN-ODE loop (backwards = oldest-first in this reversed array)
    h = np.zeros((B, H), dtype=np.float32)
    for t in range(T - 1, 0, -1):
        t0 = np.full(B, t_sc[t],   dtype=np.float32)
        t1 = np.full(B, t_sc[t-1], dtype=np.float32)
        same = (t1 == t0)
        t1_adj = t1 - DT * same
        (h,) = ode.run(None, {"h": h, "t0": t0, "t1": t1_adj})
        (h,) = gru.run(None, {
            "x_embedded": x_emb[t-1],
            "h":          h,
            "mask":       mask[t-1:t].astype(bool),
        })

    # Aggregation: evolve h over fixed time grid
    t_grid = np.linspace(0, T_MAX, N_POINTS, dtype=np.float32) * T_SCALE
    zs = [h]
    for i in range(1, N_POINTS):
        t0 = np.full(B, t_grid[i-1], dtype=np.float32)
        t1 = np.full(B, t_grid[i],   dtype=np.float32)
        (h,) = ode.run(None, {"h": h, "t0": t0, "t1": t1})
        zs.append(h)

    zs = np.stack(zs, axis=1)                           # (B, N, H)
    t_col = np.broadcast_to(t_grid[None, :, None], (B, N_POINTS, 1))
    zs_t = np.concatenate([zs, t_col], axis=-1)         # (B, N, H+1)
    (z,) = deepset.run(None, {"zs_t": zs_t.astype(np.float32)})
    return z  # (B, H) = (1, 128)
```

## Preprocessing steps

1. **Band mapping**: map photometric filter names to integer indices 0–5
   (the exact mapping used during training must be preserved)
2. **Flux normalization**: min-max normalization per light curve (subtract min,
   divide by range), applied identically to the training data loader
3. **Time normalization**: standardize (subtract mean, divide by std) per object
4. **Sort descending by time** before passing to the glue loop
5. **Pad** to the same sequence length within a batch; set `mask=False` for
   padded positions

## Weights

Source: TBD (checkpoint directory from SELDON training run)

Dataset: ELAsTiCC, SNIa class, 6 photometric bands (u, g, r, i, z, y),
~1M light curves, ~30 observations per object on average.

## Export

```bash
uv run --project models/seldon python -m seldon_prep export \
    --checkpoint /path/to/logs/SELDON_run/version_0 \
    --out-dir models/seldon/out/onnx
```
