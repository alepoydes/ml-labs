"""Episode diagnostics shared by command-line tables, viewers and reports."""

import math

EPISODE_COLUMNS = {
    "case": "Episode",
    "score": "Points",
    "status": "Outcome",
    "progress": "Accepted (m)",
    "travelled_distance": "Travelled (m)",
    "completion_fraction": "Completed (%)",
    "route_length": "Required (m)",
    "time": "Time (s)",
    "leg": "Leg",
}
OUTCOMES = {
    "running": "Running",
    "finished": "Finished",
    "fallen": "Fall",
    "timeout": "Time limit",
    "interrupted": "Interrupted; no result",
}


def episode_metrics(info, row=0):
    """Snapshot judge diagnostics, without observations or private course data."""
    result = {
        key: float(info[key][row])
        for key in (
            "score",
            "time",
            "progress",
            "route_length",
            "completion_fraction",
            "travelled_distance",
            "reference_time",
            "outward_time",
        )
    }
    result["outward_time"] = (
        result["outward_time"] if math.isfinite(result["outward_time"]) else None
    )
    return result | {
        "status": str(info["status"][row]),
        "leg": "return" if info["leg"][row] else "outward",
    }


def display(key, value):
    if value is None:
        return "—"
    if key == "status":
        return OUTCOMES.get(value, str(value))
    if key in {"completion_fraction", "mean_completion_fraction"}:
        return f"{100 * value:.1f}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def text_table(rows):
    cells = [list(EPISODE_COLUMNS.values())] + [
        [display(key, row.get(key)) for key in EPISODE_COLUMNS] for row in rows
    ]
    widths = [max(map(len, column)) for column in zip(*cells)]
    return "\n".join(
        " | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in cells
    )
