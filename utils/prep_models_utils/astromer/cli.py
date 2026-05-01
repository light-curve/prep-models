from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path


def run_main(package_name: str) -> None:
    config_module = import_module(f"{package_name}.config")
    model_dir = Path(config_module.MODEL_DIR)
    default_onnx_dir = model_dir / "out" / "onnx"
    default_test_data_dir = model_dir / "out" / "test-data"

    parser = argparse.ArgumentParser(prog=package_name)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Initialize git submodule.")
    p_fetch.add_argument(
        "--update",
        action="store_true",
        help="Advance the submodule to the latest upstream remote commit.",
    )
    p_dl = sub.add_parser(
        "download", help="Download pretrained weights and survey records."
    )
    p_dl.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data is already present.",
    )

    p_export = sub.add_parser("export", help="Export to ONNX.")
    p_export.add_argument("--output-dir", type=Path, default=default_onnx_dir)

    p_td = sub.add_parser("test-data", help="Generate test parquet.")
    p_td.add_argument("--output-dir", type=Path, default=default_test_data_dir)
    p_td.add_argument("--n-samples", type=int, default=10)

    args = parser.parse_args()

    if args.command == "fetch":
        run_fetch = import_module(f"{package_name}.fetch").run_fetch
        run_fetch(update=args.update)
    elif args.command == "download":
        run_download = import_module(f"{package_name}.download").run_download
        run_download(force=args.force)
    elif args.command == "export":
        run_export = import_module(f"{package_name}.export").run_export
        run_export(args.output_dir)
    elif args.command == "test-data":
        run_test_data = import_module(f"{package_name}.test_data").run_test_data
        run_test_data(args.output_dir, args.n_samples)
