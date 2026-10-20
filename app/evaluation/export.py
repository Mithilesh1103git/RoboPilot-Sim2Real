"""Export JSON Lines trial records to CSV."""

import csv
import json
from pathlib import Path


def export_jsonl_to_csv(source: str, destination: str) -> None:
    rows = [
        json.loads(line)
        for line in Path(source).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("No trial records found.")
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
