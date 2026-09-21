# AI Agent guide for prep-models

## Package structure

Three tiers, each with isolated dependencies:

```
prep_models/        Root CLI (Python 3.14, typer). Dispatches via `uv run`.
utils/              Shared lib (Python ≥ 3.10, requests/tqdm/pyarrow).
models/<name>/      One sub-package per model. Own Python pin + uv.lock.
```

## Vendored upstream code

Never change vendored code inside git submodules. In this repository, upstream model
code lives under `models/<name>/code/` (for example `models/astromer1/code/` and
`models/astromer2/code/`). Treat those directories as read-only reference code:
inspect them when needed, but put all integration, patches, wrappers, conversion
logic, tests, and documentation in the surrounding non-submodule package files.

The root CLI never imports ML libraries. It calls model sub-packages via:
```python
subprocess.run(
    [
        "uv",
        "run",
        "--project",
        "models/<name>",
        "python",
        "-m",
        "<name>_prep",
        command,
        ...,
    ]
)
```

## HuggingFace publishing

ONNX files are published to the [light-curve](https://huggingface.co/light-curve) HuggingFace organization.
Each model gets its own repository named after the model (e.g. `light-curve/astromer2`).

The HuggingFace repository must:
- Use the **same license** as the original upstream model
- Include a `README.md` (the HuggingFace model card) with full paper citation, model description, input/output format, and preprocessing steps

To upload:
```bash
# Install huggingface_hub in the root env (add to pyproject.toml if making this a command)
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='models/astromer2/out/onnx/',
    repo_id='light-curve/astromer2',
    repo_type='model',
)"
```

---

## Adding a new model

1. **Add git submodule**
   ```bash
   git submodule add <upstream_url> models/<name>/code
   ```

2. **Create model directory structure**
   ```
   models/<name>/
   ├── README.md           (see per-model README template below)
   ├── code/               (git submodule; read-only vendored upstream code)
   ├── LICENSE             (copy from upstream)
   ├── weights/            (empty; populated by download command)
   ├── .python-version     (pin appropriate Python version for the ML framework)
   ├── pyproject.toml      (our deps — do NOT copy from upstream)
   └── <name>_prep/
       ├── __init__.py
       ├── __main__.py     (argparse: fetch|download|export|test-data)
       ├── fetch.py
       ├── download.py
       ├── export.py
       └── test_data.py
   ```

3. **Write `pyproject.toml`** with only the deps we need (ML framework, tf2onnx or torch.onnx, onnx, toml, and `prep-models-utils @ ../../utils`). Do not copy the upstream requirements.

4. **Lock the environment**
   ```bash
   cd models/<name> && uv lock
   ```

5. **Register in root CLI** — add to `_MODELS` dict in `prep_models/cli.py`:
   ```python
   "<name>": {
       "project": REPO_ROOT / "models" / "<name>",
       "module": "<name>_prep",
   },
   ```

6. **Write `README.md`** for the model (see template below).

7. **Test end-to-end**
   ```bash
   uv run prep-models <name> fetch
   uv run prep-models <name> download
   uv run prep-models <name> export
   uv run prep-models <name> test-data
   ```

## Per-model README template

```markdown
# <Model Name>

## Paper

<full citation>

```bibtex
@article{...}
```

## Original code

<GitHub/Codeberg URL> (git submodule at `models/<name>/code/`)

## License

<License name and link>

## Model overview

<1-paragraph summary of architecture>

## Inputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| input | [batch, seq_len, 1] | ... |
| times | [batch, seq_len, 1] | ... |
| mask_in | [batch, seq_len, 1] | 1=valid, 0=masked/padded |

## Outputs (ONNX)

| File | Shape | Aggregation |
|------|-------|-------------|
| <model>_mean.onnx | [batch, dim] | masked mean pool |
| <model>_max.onnx | [batch, dim] | masked max pool |
| <model>_full.onnx | [batch, seq_len, dim] | none |

## Preprocessing steps

1. ...
2. ...

## Weights

Source: <Zenodo/HuggingFace URL>
Dataset: <training dataset name and description>
```

## Development workflow

```bash
# Install root deps and pre-commit hooks
uv sync
uvx pre-commit install
# Pre-commit runs on every commit: ruff lint+format, trailing whitespace,
# end-of-file newlines, TOML/YAML validity, no large files, no merge conflicts.
# Submodule code/ directories are excluded from all hooks.
# Run manually: uvx pre-commit run --all-files

# Run a specific command without going through the root CLI
uv run --project models/astromer2 python -m astromer2_prep export

# Validate exported ONNX numerically (compare TF vs ONNX runtime)
uv run --project models/astromer2 python - <<'EOF'
import numpy as np, onnxruntime as rt, tensorflow as tf, sys
sys.path.insert(0, "models/astromer2/code")
from astromer2_prep.export import _load_model, _encoder_fn

model, _ = _load_model()
encoder = model.get_layer("encoder")
inp = tf.random.normal([2, 200, 1])
times = tf.random.uniform([2, 200, 1], 50000, 60000)
mask = tf.ones([2, 200, 1])

fn = _encoder_fn(encoder, "mean")
tf_out = fn(inp, times, mask).numpy()

sess = rt.InferenceSession("models/astromer2/out/onnx/astromer2_mean.onnx")
onnx_out = sess.run(None, {"input": inp.numpy(), "times": times.numpy(), "mask_in": mask.numpy()})[0]

print("Max abs diff:", np.abs(tf_out - onnx_out).max())
assert np.allclose(tf_out, onnx_out, atol=1e-4), "Mismatch!"
print("OK")
EOF
```
