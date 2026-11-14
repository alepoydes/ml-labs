from importlib import import_module

from coursekit.common import CourseError

MODULES = {
    "blindjack": "blindjack",
    "line_following": "line_following_task",
    "visual_queries": "visual_queries",
}


def require_lab(lab: str, api: int | None = None) -> None:
    if (
        not isinstance(lab, str)
        or lab not in MODULES
        or api is not None
        and (
            type(api) is not int or api not in ((3,) if lab == "visual_queries" else (1, 2))
        )
    ):
        raise CourseError(f"Unsupported lab/API: {lab}/{api}; update the course files.")


def get_lab(lab: str):
    require_lab(lab)
    return import_module(f"coursekit.labs.{MODULES[lab]}")
