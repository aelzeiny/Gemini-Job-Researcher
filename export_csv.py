#!/usr/bin/env python3
"""
Export job summaries to a CSV file.

Reads all complete summary dirs under outputs/(model)/summaries/ and writes
a single CSV to outputs/(model)/jobs.csv. When multiple summary dirs exist
for the same job title, the most recently generated one is used.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


CATEGORIES = ["wlb", "stress", "burnout", "morning_hours", "thinking_style"]

OVERALL_ORDER = ["Highly Recommended", "Recommended", "Neutral", "Not Recommended", "Avoid"]


def extract_job_title(dir_name: str) -> str:
    title = re.sub(r"_\d{8}_\d{6}$", "", dir_name)
    return title.replace("_", " ")


def is_complete(summary_dir: Path) -> bool:
    return all((summary_dir / f"{cat}.json").exists() for cat in CATEGORIES + ["overall"])


def load_summary(summary_dir: Path) -> dict:
    data = {}
    for cat in CATEGORIES:
        cat_data = json.loads((summary_dir / f"{cat}.json").read_text(encoding="utf-8"))
        data[f"{cat}_rating"] = cat_data.get("rating", "")
    overall = json.loads((summary_dir / "overall.json").read_text(encoding="utf-8"))
    data["overall_rating"] = overall.get("overall_rating", "")
    data["summary"] = overall.get("summary", "")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Export job summaries to outputs/(model)/jobs.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model subdirectory to export from (default: gemini-3-flash-preview)",
    )
    args = parser.parse_args()

    model_safe = args.model.replace("/", "-").replace(":", "-")
    summaries_dir = Path(args.output_dir) / model_safe / "summaries"

    if not summaries_dir.exists():
        print(f"Error: Summaries directory not found: {summaries_dir}")
        sys.exit(1)

    # Collect complete summary dirs grouped by job title
    by_title: dict[str, list[Path]] = {}
    incomplete = 0
    for entry in sorted(summaries_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not is_complete(entry):
            incomplete += 1
            continue
        title = extract_job_title(entry.name)
        by_title.setdefault(title, []).append(entry)

    if not by_title:
        print("No complete summaries found.")
        sys.exit(0)

    # For each title pick the most recent dir (timestamp suffix in name)
    rows = []
    duplicates = 0
    for title, dirs in by_title.items():
        latest = sorted(dirs, key=lambda d: d.name)[-1]
        duplicates += len(dirs) - 1
        rows.append({"job_title": title, **load_summary(latest)})

    # Sort by overall rating order, then alphabetically within each tier
    def sort_key(row):
        rating = row["overall_rating"]
        tier = OVERALL_ORDER.index(rating) if rating in OVERALL_ORDER else len(OVERALL_ORDER)
        return (tier, row["job_title"])

    rows.sort(key=sort_key)

    # Write CSV
    csv_path = Path(args.output_dir) / model_safe / "_jobs.csv"
    fieldnames = (
        ["job_title", "overall_rating"]
        + [f"{cat}_rating" for cat in CATEGORIES]
        + ["summary"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} jobs → {csv_path}")
    if duplicates:
        print(f"Skipped {duplicates} older duplicate summary dir(s)")
    if incomplete:
        print(f"Skipped {incomplete} incomplete summary dir(s) (missing one or more category JSONs)")


if __name__ == "__main__":
    main()
