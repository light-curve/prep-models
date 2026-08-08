"""Upload ONNX files, test data, README, and LICENSE to HuggingFace."""

from pathlib import Path


def run_upload(
    model_name: str,
    hf_repo: str,
    model_dir: Path,
    onnx_dir: Path,
    token: str | None,
    create_repo: bool,
) -> None:
    from huggingface_hub import HfApi

    model_dir = Path(model_dir)
    onnx_dir = Path(onnx_dir)
    api = HfApi(token=token)

    if create_repo:
        api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
        print(f"Repository {hf_repo} ready.")

    # Collect ONNX files
    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No ONNX files found in {onnx_dir}. Run 'export' first."
        )

    # Collect metadata files from the model directory
    meta_files: list[tuple[Path, str]] = []
    for name in ("README.md", "LICENSE"):
        p = model_dir / name
        if p.exists():
            meta_files.append((p, name))

    print(f"Uploading to {hf_repo}:")
    for f in onnx_files:
        print(f"  {f.name}")
    for _, dest in meta_files:
        print(f"  {dest}")

    for local_path in onnx_files:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=hf_repo,
            repo_type="model",
        )

    for local_path, dest_name in meta_files:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=dest_name,
            repo_id=hf_repo,
            repo_type="model",
        )

    print(f"Done. View at: https://huggingface.co/{hf_repo}")
