"""Upload ONNX files and test data to HuggingFace."""

from pathlib import Path
from typing import Optional


def run_upload(
    model_name: str,
    hf_repo: str,
    onnx_dir: Path,
    test_data_dir: Optional[Path],
    token: Optional[str],
    create_repo: bool,
) -> None:
    from huggingface_hub import HfApi

    onnx_dir = Path(onnx_dir)
    api = HfApi(token=token)

    if create_repo:
        api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
        print(f"Repository {hf_repo} ready.")

    # Collect files to upload
    onnx_files = sorted(onnx_dir.glob(f"{model_name}_*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No ONNX files matching '{model_name}_*.onnx' found in {onnx_dir}. "
            "Run 'export' first."
        )

    parquet_files: list[Path] = []
    if test_data_dir is not None:
        parquet_files = sorted(Path(test_data_dir).glob(f"{model_name}_*.parquet"))

    all_files = onnx_files + parquet_files
    print(f"Uploading {len(all_files)} file(s) to {hf_repo}:")
    for f in all_files:
        print(f"  {f.name}")

    for local_path in all_files:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=hf_repo,
            repo_type="model",
        )

    print(f"Done. View at: https://huggingface.co/{hf_repo}")
