"""Entry point: uv run python -m atat_prep <command> [options]"""

from prep_models_utils import run_main


def main() -> None:
    run_main("atat_prep")


if __name__ == "__main__":
    main()
