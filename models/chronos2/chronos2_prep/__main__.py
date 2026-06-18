"""Entry point: uv run python -m chronos2_prep <command> [options]"""

from prep_models_utils.cli import run_main


def main() -> None:
    run_main("chronos2_prep")


if __name__ == "__main__":
    main()
