from coursekit.common import CourseError


def require_lab(lab: str, api: int = 1) -> None:
    if lab != "blindjack" or type(api) is not int or api != 1:
        raise CourseError(f"Unsupported lab/API: {lab}/{api}; update the course files.")
