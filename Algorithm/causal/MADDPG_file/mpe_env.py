"""Import MPE environments (mpe2 for PettingZoo >= 1.26, legacy pettingzoo.mpe otherwise)."""
import importlib


def import_mpe_module(env_name: str):
    last_err = None
    for pkg in ("mpe2", "pettingzoo.mpe"):
        try:
            return importlib.import_module(f"{pkg}.{env_name}")
        except ModuleNotFoundError as exc:
            last_err = exc
    raise ModuleNotFoundError(
        f"Cannot import MPE env '{env_name}'. "
        "Install: pip install mpe2 pettingzoo gymnasium"
    ) from last_err
