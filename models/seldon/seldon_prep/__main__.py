"""Entry point: uv run --project models/seldon python -m seldon_prep <command>"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="SELDON ONNX prep utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export encoder to ONNX")
    exp.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to Lightning checkpoint directory (contains hparams.yaml and checkpoints/)",
    )
    exp.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/seldon/out/onnx"),
        help="Output directory for ONNX files",
    )
    exp.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.command == "export":
        from seldon_prep.export import run_export

        run_export(args.out_dir, args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
