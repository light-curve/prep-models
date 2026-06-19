"""Entry point: uv run python -m chronos_bolt_prep <command> --size <size> [opts]

Unlike the single-model packages, Chronos-Bolt ships four sizes from one
package, so every command takes a required ``--size`` (tiny/mini/small/base)
and reads/writes a per-size ``out/<size>/`` subdirectory.
"""

from __future__ import annotations

import argparse
from importlib import import_module

from chronos_bolt_prep.config import DEFAULT_SIZE, SIZES, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="chronos_bolt_prep")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_size(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--size",
            choices=sorted(SIZES),
            default=DEFAULT_SIZE,
            help="Chronos-Bolt size to operate on.",
        )

    p_fetch = sub.add_parser("fetch", help="No-op (no code submodule).")
    _add_size(p_fetch)
    p_fetch.add_argument("--update", action="store_true")

    p_dl = sub.add_parser("download", help="Download pretrained weights.")
    _add_size(p_dl)
    p_dl.add_argument("--force", action="store_true")

    p_export = sub.add_parser("export", help="Export to ONNX.")
    _add_size(p_export)
    p_export.add_argument("--output-dir", default=None)

    p_td = sub.add_parser("test-data", help="Generate test parquet.")
    _add_size(p_td)
    p_td.add_argument("--output-dir", default=None)
    p_td.add_argument("--n-samples", type=int, default=10)

    args = parser.parse_args()
    size = args.size

    if args.command == "fetch":
        import_module("chronos_bolt_prep.fetch").run_fetch(
            update=args.update, size=size
        )
    elif args.command == "download":
        import_module("chronos_bolt_prep.download").run_download(
            force=args.force, size=size
        )
    elif args.command == "export":
        output_dir = args.output_dir or out_dir(size) / "onnx"
        import_module("chronos_bolt_prep.export").run_export(output_dir, size)
    elif args.command == "test-data":
        output_dir = args.output_dir or out_dir(size) / "test-data"
        import_module("chronos_bolt_prep.test_data").run_test_data(
            output_dir, args.n_samples, size=size
        )


if __name__ == "__main__":
    main()
