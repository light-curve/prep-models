import os
import subprocess
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    name="prep-models",
    help="Convert open-weight light-curve models to ONNX.",
    no_args_is_help=True,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_ORG = "light-curve"

_MODELS: dict = {
    "astromer1": {
        "project": REPO_ROOT / "models" / "astromer1",
        "module": "astromer1_prep",
        "hf_repo": f"{HF_ORG}/astromer1",
    },
    "astromer2": {
        "project": REPO_ROOT / "models" / "astromer2",
        "module": "astromer2_prep",
        "hf_repo": f"{HF_ORG}/astromer2",
    },
    "atat": {
        "project": REPO_ROOT / "models" / "atat",
        "module": "atat_prep",
        "hf_repo": f"{HF_ORG}/atat",
    },
    "atcat": {
        "project": REPO_ROOT / "models" / "atcat",
        "module": "atcat_prep",
        "hf_repo": f"{HF_ORG}/atcat",
        "onnx_transformations": ["bf16_to_f32"],
    },
    "astrom3": {
        "project": REPO_ROOT / "models" / "astrom3",
        "module": "astrom3_prep",
        "hf_repo": f"{HF_ORG}/astrom3",
    },
    "astronet-tinho": {
        "project": REPO_ROOT / "models" / "astronet-tinho",
        "module": "astronet_tinho_prep",
        "hf_repo": f"{HF_ORG}/astronet-tinho",
    },
}


def _run(model: str, command: str, extra: List[str]) -> None:
    info = _MODELS[model]
    # Drop VIRTUAL_ENV so uv doesn't warn that the parent venv doesn't match
    # the sub-project's own venv.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(info["project"]),
            "python",
            "-m",
            info["module"],
            command,
            *extra,
        ],
        check=True,
        env=env,
    )


def _validate_onnx(onnx_dir: Path) -> None:
    import numpy as np
    import onnxruntime as rt

    _NP_DTYPE = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
        "tensor(float16)": np.float16,
    }

    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        typer.echo(f"Warning: no .onnx files found in {onnx_dir}", err=True)
        return
    skipped: list[str] = []
    for path in onnx_files:
        typer.echo(f"Validating {path.name} with onnxruntime ...")
        is_bf16 = path.stem.endswith("_bf16")
        try:
            sess = rt.InferenceSession(str(path))
        except Exception as e:
            if is_bf16 and "NOT_IMPLEMENTED" in str(e):
                typer.echo("  Skipped: ORT CPU provider does not support bfloat16 ops.")
                skipped.append(path.name)
                continue
            raise

        feeds = {}
        for inp in sess.get_inputs():
            dtype = _NP_DTYPE.get(inp.type, np.float32)
            # Replace dynamic (None/0/string) dims: first dim → batch=2, rest → 1
            shape = [
                2 if i == 0 else (d if isinstance(d, int) and d > 0 else 1)
                for i, d in enumerate(inp.shape)
            ]
            typer.echo(f"  input  '{inp.name}': {inp.shape}  → feed shape {shape}")
            feeds[inp.name] = np.zeros(shape, dtype=dtype)

        outputs = sess.run(None, feeds)
        for out_meta, out_arr in zip(sess.get_outputs(), outputs):
            typer.echo(f"  output '{out_meta.name}': {list(out_arr.shape)}")
        typer.echo(f"  OK: {path.name}")

    if skipped:
        typer.echo(
            f"\nWARNING: {len(skipped)} bfloat16 model(s) were not validated"
            f" ({', '.join(skipped)}) — ORT CPU provider does not support bfloat16 ops."
            f' Please validate on a CUDA machine using providers=["CUDAExecutionProvider"].',
            err=True,
        )


def _model_app(model_name: str) -> typer.Typer:
    sub = typer.Typer(help=f"Commands for {model_name}.", no_args_is_help=True)

    @sub.command("fetch")
    def fetch(
        update: bool = typer.Option(
            False,
            "--update",
            help="Advance the submodule to the latest upstream remote commit.",
        ),
    ) -> None:
        """Initialize the pinned git submodule, or update it when requested."""
        extra = ["--update"] if update else []
        _run(model_name, "fetch", extra)

    @sub.command("download")
    def download() -> None:
        """Download pretrained weights."""
        _run(model_name, "download", [])

    @sub.command("export")
    def export(
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Directory for ONNX files (default: models/<name>/out/onnx/).",
        ),
    ) -> None:
        """Export model to ONNX (one file per aggregation variant)."""
        extra = ["--output-dir", str(output_dir)] if output_dir is not None else []
        _run(model_name, "export", extra)

        onnx_dir = output_dir or (REPO_ROOT / "models" / model_name / "out" / "onnx")

        for transform in _MODELS[model_name].get("onnx_transformations", []):
            if transform == "bf16_to_f32":
                from prep_models.onnx_utils import convert_bf16_to_f32

                for bf16_path in sorted(onnx_dir.glob("*_bf16.onnx")):
                    f32_path = bf16_path.with_stem(
                        bf16_path.stem.removesuffix("_bf16") + "_f32"
                    )
                    typer.echo(f"Converting {bf16_path.name} → {f32_path.name} ...")
                    convert_bf16_to_f32(bf16_path, f32_path)
            else:
                raise ValueError(f"Unknown ONNX transformation: {transform!r}")

        _validate_onnx(onnx_dir)

    @sub.command("test-data")
    def test_data(
        output_dir: Optional[Path] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Directory for parquet files (default: models/<name>/out/test-data/).",
        ),
        n_samples: int = typer.Option(
            10, "--n-samples", "-n", help="Number of light curves to include."
        ),
    ) -> None:
        """Generate test data: original inputs + model outputs as parquet."""
        extra = ["--output-dir", str(output_dir)] if output_dir is not None else []
        _run(model_name, "test-data", [*extra, "--n-samples", str(n_samples)])

    @sub.command("upload")
    def upload(
        onnx_dir: Optional[Path] = typer.Option(
            None,
            "--onnx-dir",
            "-i",
            help="Directory containing ONNX files (default: models/<name>/out/onnx/).",
        ),
        token: Optional[str] = typer.Option(
            None,
            "--token",
            envvar="HF_TOKEN",
            help="HuggingFace API token (or set HF_TOKEN).",
        ),
        create_repo: bool = typer.Option(
            False, "--create-repo", help="Create HuggingFace repo if it doesn't exist."
        ),
    ) -> None:
        """Upload ONNX files, README, and LICENSE to HuggingFace."""
        from prep_models.upload import run_upload

        hf_repo = _MODELS[model_name]["hf_repo"]
        model_dir = _MODELS[model_name]["project"]
        run_upload(
            model_name=model_name,
            hf_repo=hf_repo,
            model_dir=model_dir,
            onnx_dir=onnx_dir or model_dir / "out" / "onnx",
            token=token,
            create_repo=create_repo,
        )

    return sub


for _name in _MODELS:
    app.add_typer(_model_app(_name), name=_name)
