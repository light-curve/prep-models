# Refactor Astromer Shared Logic

## Summary
Extract the materially duplicated Astromer implementation into `utils/prep_models_utils/astromer/`, while keeping `models/astromer1` and `models/astromer2` as separate `uv` projects with separate dependencies, package names, and version-specific model-loading behavior.

Success criteria:
- `astromer1_prep` and `astromer2_prep` still expose the same commands and outputs.
- Shared logic is centralized in `prep_models_utils.astromer`.
- Version packages become thin wrappers around shared helpers plus version-specific configuration/callbacks.
- No repo layout rename, no environment merge, no top-level CLI redesign in this refactor.

## Key Changes

### Shared Astromer package
Add `utils/prep_models_utils/astromer/` with a compact shared API:

- `AstromerConfig` dataclass or typed dict with:
  - `model_name`
  - `code_dir`
  - `weights_dir`
  - `zenodo_record_id`
  - `zenodo_key`
  - `output_prefix`
  - `test_data_filename`
- Shared helper functions for:
  - adding vendored `code/` to `sys.path`
  - generating synthetic light curves
  - preprocessing via upstream `format_inp_astromer`
  - running encoder inference and masked mean pooling
  - building ONNX wrapper functions for `mean`, `max`, `full`
  - executing the common ONNX export loop
  - downloading weights from Zenodo into a target directory

Structure the shared package as three modules:
- `prep_models_utils.astromer.common`: config, path/import helpers, synthetic/test-data helpers, shared download helper
- `prep_models_utils.astromer.export`: shared encoder wrapper and ONNX export flow
- `prep_models_utils.astromer.test_data`: shared parquet test-data flow

Re-export only the stable entry points from `prep_models_utils.astromer.__init__`.

### Version wrappers
Refactor both model packages so they only retain version-specific behavior.

For `weights.py` in each version:
- Keep only the Zenodo constants and local `MODEL_DIR`/`WEIGHTS_DIR`.
- Build `AstromerConfig`.
- Call shared `download_weights(config)`.

For `test_data.py` in each version:
- Keep only local path constants and a version-specific import of `_load_model`.
- Build `AstromerConfig`.
- Call shared `run_test_data(config, output_dir, n_samples, load_model=_load_model)`.

For `export.py` in each version:
- Keep version-specific `MODEL_DIR`, `CODE_DIR`, `WEIGHTS_DIR`.
- Keep version-specific `_find_weights_dir()`:
  - v1: locate nested `*/config.toml` under `weights/`
  - v2: require `weights/config.toml`
- Keep version-specific `_load_model()` using `presentation.pipelines.steps.model_design.build_model`.
- Replace duplicated encoder-function construction and ONNX export loop with a shared `run_export(config, output_dir, load_model=_load_model)` helper.

For `fetch.py` in each version:
- Keep only `SUBMODULE_PATH`.
- Delegate to existing shared `prep_models_utils.run_fetch`.

Leave `__main__.py` behavior unchanged except for any import adjustments required by slimmer wrapper modules.

### Public/internal interfaces
Add these shared function signatures and keep them stable across both versions:

```python
@dataclass
class AstromerConfig:
    model_name: str
    code_dir: Path
    weights_dir: Path
    zenodo_record_id: str
    zenodo_key: str
    output_prefix: str
    test_data_filename: str


def download_weights(config: AstromerConfig) -> None: ...
def run_test_data(
    config: AstromerConfig,
    output_dir: Path,
    n_samples: int,
    *,
    load_model: Callable[[], tuple[object, object]],
) -> None: ...
def run_export(
    config: AstromerConfig,
    output_dir: Path,
    *,
    load_model: Callable[[], tuple[object, object]],
) -> None: ...
```

Internal shared helpers may remain private and should not be re-exported from `prep_models_utils`.

## Test Plan
Run non-mutating validation after the refactor:

- `python3 -m compileall utils/prep_models_utils prep_models models/astromer1/astromer1_prep models/astromer2/astromer2_prep`
- Verify `fetch --help` still works through:
  - root CLI for each model
  - per-model `python -m ..._prep fetch --help`
- Verify imports resolve for both wrappers without circular dependencies.
- Check that shared export logic still produces filenames:
  - `astromer1_mean.onnx`, `astromer1_max.onnx`, `astromer1_full.onnx`
  - `astromer2_mean.onnx`, `astromer2_max.onnx`, `astromer2_full.onnx`
- Check that shared test-data logic still writes:
  - `astromer1_test.parquet`
  - `astromer2_test.parquet`

If full `uv run` validation is blocked by sandbox/network/dependency resolution, treat `compileall` plus help-path verification as the minimum acceptance gate and note the remaining runtime verification gap.

## Assumptions
- Keep separate package names: `astromer1_prep` and `astromer2_prep`.
- Keep separate `pyproject.toml` and `uv.lock` files.
- Keep current output naming and HuggingFace naming unchanged.
- Do not attempt to share `_load_model()` or `_find_weights_dir()` beyond passing them into shared helpers.
- Do not refactor README/docs in the same change unless imports/commands become inaccurate as a direct consequence of the code refactor.
