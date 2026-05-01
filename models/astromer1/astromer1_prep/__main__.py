"""Entry point: uv run python -m astromer1_prep <command> [options]"""

from prep_models_utils.astromer import run_main


def main() -> None:
    run_main("astromer1_prep")


if __name__ == "__main__":
    main()
