"""Aggregate task-trial metrics."""

from statistics import mean


def summarize_trials(rows: list[dict]) -> dict:
    completed = [row for row in rows if isinstance(row.get("success"), bool)]
    if not completed:
        return {
            "completed_trials": 0,
            "success_rate": None,
            "recovery_rate": None,
            "mean_duration_seconds": None,
        }
    recovery_rows = [row for row in completed if isinstance(row.get("recovered"), bool)]
    durations = [
        float(row["duration_seconds"])
        for row in completed
        if isinstance(row.get("duration_seconds"), (int, float))
    ]
    return {
        "completed_trials": len(completed),
        "success_rate": sum(row["success"] for row in completed) / len(completed),
        "recovery_rate": (
            sum(row["recovered"] for row in recovery_rows) / len(recovery_rows)
            if recovery_rows
            else None
        ),
        "mean_duration_seconds": mean(durations) if durations else None,
    }
