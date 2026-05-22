"""Entry point: uv run python -m astra_clr_prep <command> [options]"""

from prep_models_utils.cli import run_main


def main() -> None:
    run_main("astra_clr_prep")


if __name__ == "__main__":
    main()
