"""Metrics helpers for evaluation records."""


def summarize_trials(rows: list[dict]) -> dict:
    completed = [r for r in rows if isinstance(r.get("success"), bool)]
    if not completed:
        return {"completed_trials": 0, "success_rate": None, "recovery_rate": None}
    successes = sum(r["success"] for r in completed)
    recovery_rows = [r for r in completed if isinstance(r.get("recovered"), bool)]
    return {
        "completed_trials": len(completed),
        "success_rate": successes / len(completed),
        "recovery_rate": (
            sum(r["recovered"] for r in recovery_rows) / len(recovery_rows)
            if recovery_rows else None
        ),
    }
