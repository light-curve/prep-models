from prep_models_utils import run_main


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        from astrom3_prep.validate import run_validate

        run_validate()
    else:
        run_main("astrom3_prep")


if __name__ == "__main__":
    main()
