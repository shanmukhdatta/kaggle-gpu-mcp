"""Kaggle CPU stage: combine CSV input datasets or generate deterministic demo data.

For real datasets, attached CSV inputs must share column names and compatible types.
The trainer example expects columns f1, f2, label (0 or 1).
"""
import csv
import glob
import json
import random
from pathlib import Path

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
out.mkdir(exist_ok=True)
files = sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True))
rows = []
fields = None
for candidate in files:
    with open(candidate, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            continue
        if fields is None:
            fields = reader.fieldnames
        if reader.fieldnames != fields:
            continue
        rows.extend(reader)
if not rows:
    # Demo-only synthetic data. Never represent synthetic data as a real collected dataset.
    fields = ["f1", "f2", "label"]
    rng = random.Random(43)
    for _ in range(400):
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        rows.append({"f1": round(x, 5), "f2": round(y, 5), "label": int(x + y > 0)})
unique = {}
for row in rows:
    unique[tuple(row.get(name, "") for name in fields)] = row
with (out / "merged.csv").open("w", newline="", encoding="utf-8") as dest:
    writer = csv.DictWriter(dest, fieldnames=fields)
    writer.writeheader()
    writer.writerows(unique.values())
(out / "preparation.json").write_text(json.dumps({
    "source_csv_count": len(files), "rows_in": len(rows),
    "rows_out": len(unique), "columns": fields,
    "source": "kaggle-input-or-synthetic-demo",
}, indent=2), encoding="utf-8")
print(f"Prepared {len(unique)} rows from {len(files)} attached CSV input files")
