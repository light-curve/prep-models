"""Entry point: uv run python -m astromer2_prep <command> [options]"""

from prep_models_utils.astromer import run_main


def main() -> None:
    run_main("astromer2_prep")


if __name__ == "__main__":
    main()
