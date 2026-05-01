# prep-models

Tools for exporting open-weight astronomical light-curve embedding models to ONNX.

ONNX files are published to the [light-curve](https://huggingface.co/light-curve) HuggingFace organization, one repository per model.
Each HuggingFace repository carries the original model's license and a README with full citations.

## Models

| Model | Framework | HuggingFace | Status |
|-------|-----------|-------------|--------|
| [Astromer 1](models/astromer1/README.md) | TensorFlow 2.14 | [light-curve/astromer1](https://huggingface.co/light-curve/astromer1) | implemented |
| [Astromer 2](models/astromer2/README.md) | TensorFlow 2.14 | [light-curve/astromer2](https://huggingface.co/light-curve/astromer2) | implemented |

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
```

Or equivalently:

```bash
uv run python -m prep_models astromer2 export
```

## Export outputs

Each `export` command produces three ONNX files, one per aggregation strategy:

| File | Aggregation | Output shape |
|------|-------------|--------------|
| `<model>_mean.onnx` | Masked mean pooling | `[batch, embedding_dim]` |
| `<model>_max.onnx` | Masked max pooling | `[batch, embedding_dim]` |
| `<model>_full.onnx` | No pooling (full sequence) | `[batch, seq_len, embedding_dim]` |

## Development

```bash
uv sync          # install root CLI deps
uv run prep-models --help
```
