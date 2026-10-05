from importlib import import_module

from coursekit.common import CourseError

MODULES = {"blindjack": "blindjack", "line_following": "line_following_task"}


def require_lab(lab: str, api: int = 1) -> None:
    if not isinstance(lab, str) or lab not in MODULES or type(api) is not int or api != 1:
        raise CourseError(f"Unsupported lab/API: {lab}/{api}; update the course files.")


def get_lab(lab: str):
    require_lab(lab)
    return import_module(f"coursekit.labs.{MODULES[lab]}")
