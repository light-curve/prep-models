"""Entry point: uv run python -m atcat_prep <command> [options]."""

from prep_models_utils import run_main


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        from atcat_prep.validate import run_validate

        run_validate()
    else:
        run_main("atcat_prep")


if __name__ == "__main__":
    main()
