from prep_models_utils import run_fetch as run_shared_fetch

SUBMODULE_PATH = "models/astrom3/code"


def run_fetch(*, update: bool = False) -> None:
    run_shared_fetch(SUBMODULE_PATH, update=update)
