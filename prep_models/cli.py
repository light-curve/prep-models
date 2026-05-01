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
