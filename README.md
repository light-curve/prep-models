# prep-models

Tools for exporting open-weight astronomical light-curve embedding models to ONNX.

ONNX files are published to the [light-curve](https://huggingface.co/light-curve) HuggingFace organization, one repository per model.
Each HuggingFace repository carries the original model's license and a README with full citations.

## Models

| Model | Framework | HuggingFace | Status |
|-------|-----------|-------------|--------|
| [Astromer 1](models/astromer1/README.md) | TensorFlow 2.14 | [light-curve/astromer1](https://huggingface.co/light-curve/astromer1) | implemented |
| [Astromer 2](models/astromer2/README.md) | TensorFlow 2.14 | [light-curve/astromer2](https://huggingface.co/light-curve/astromer2) | implemented |
| [ATAT](models/atat/README.md) | PyTorch | pending (no upstream license — see [alercebroker/ATAT#2](https://github.com/alercebroker/ATAT/issues/2)) | implemented |
| [ATCAT](models/atcat/README.md) | PyTorch | [light-curve/atcat](https://huggingface.co/light-curve/atcat) | implemented |
| [AstroM3](models/astrom3/README.md) | PyTorch | [light-curve/astrom3](https://huggingface.co/light-curve/astrom3) | implemented |

## Architecture

The repo uses a three-tier isolated structure:

- **`utils/`** — shared library (`prep_models_utils`): download helpers, parquet I/O, abstract base classes. Python ≥ 3.10, minimal deps.
- **`models/<name>/`** — one isolated sub-package per model with its own Python version, locked environment (`uv.lock`), and dependencies we define independently of the upstream model.
- **Root** (`prep_models/`) — Typer CLI orchestrator. Dispatches to model sub-packages via `uv run --project`.

## Usage

All commands are run via `uv run`. No manual environment activation is needed.

```bash
# Astromer 2
uv run prep-models astromer2 fetch         # pull latest upstream code
uv run prep-models astromer2 download      # download pretrained weights
uv run prep-models astromer2 export        # export ONNX files
uv run prep-models astromer2 test-data --n-samples 5

# Astromer 1 (same pattern)
uv run prep-models astromer1 fetch
uv run prep-models astromer1 download
uv run prep-models astromer1 export
uv run prep-models astromer1 test-data

# ATAT (same pattern)
uv run prep-models atat fetch
uv run prep-models atat download      # also downloads one ELASTICC FITS pair for test data
uv run prep-models atat export
uv run prep-models atat test-data

# ATCAT (same pattern)
uv run prep-models atcat fetch
uv run prep-models atcat download
uv run prep-models atcat export
uv run prep-models atcat test-data
```

Or equivalently:

```bash
uv run python -m prep_models astromer2 export
```

## Export outputs

Each `export` command produces a single ONNX file (or two for ATCAT: fp32 + bf16) with multiple named outputs.
Request only the output(s) you need — onnxruntime prunes unused computation.

| Model | File(s) | Output names |
|-------|---------|--------------|
| Astromer 1 | `astromer1.onnx` | `mean`, `max`, `sequence` |
| Astromer 2 | `astromer2.onnx` | `mean`, `max`, `sequence` |
| ATAT | `atat.onnx` | `token`, `mean`, `sequence` |
| ATCAT | `atcat_bf16.onnx`, `atcat_f32.onnx` | `last`, `mean`, `sequence` |

`sequence` is the per-element transformer output (`[batch, seq_len, embedding_dim]`); for ATAT the CLS token is excluded.
`token` (ATAT only) is the CLS token — a dedicated global representation prepended to the sequence and excluded from `sequence`.
`last` (ATCAT only) is the hidden state at the last valid LC observation (`sequence[num_lc_points-1]`), a convenience slice of `sequence`.
`mean` / `max` are masked pooling over the sequence (`[batch, embedding_dim]`).

## Development

```bash
uv sync          # install root CLI deps
uv run prep-models --help
```
