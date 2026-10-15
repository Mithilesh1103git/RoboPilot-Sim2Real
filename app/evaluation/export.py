"""CSV export for JSONL trial records."""
import csv
import json
from pathlib import Path


def export_jsonl_to_csv(source: str, destination: str) -> None:
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No trial records found.")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
