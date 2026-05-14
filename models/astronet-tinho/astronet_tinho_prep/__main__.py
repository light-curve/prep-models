"""Entry point: uv run --project models/astronet-tinho python -m astronet_tinho_prep <command>"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="AstroNet-TINHO ONNX prep utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export TINHO SavedModels to ONNX")
    exp.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/astronet-tinho/out/onnx"),
        help="Output directory for ONNX files",
    )

    args = parser.parse_args()

    if args.command == "export":
        from astronet_tinho_prep.export import run_export

        run_export(args.out_dir)


if __name__ == "__main__":
    main()
