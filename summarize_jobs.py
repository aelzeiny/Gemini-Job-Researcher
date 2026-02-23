#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-genai>=1.0.0",
#   "python-dotenv>=1.0.0",
# ]
# ///
"""
Summarize job research files using Gemini.

Each job is evaluated with one focused Gemini call per category (WLB, stress,
burnout, morning hours) running in parallel, followed by a final call that
synthesizes the four ratings into an overall recommendation and summary.

Outputs are saved as JSON under outputs/(model)/summaries/.
"""

import asyncio
import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv


TEMPLATES_DIR = Path(__file__).parent / "templates"

CATEGORIES = ["wlb", "stress", "burnout", "morning_hours", "thinking_style"]

SUMMARIZE_MODEL = "gemini-3-flash-preview"


def load_prompt(filename: str) -> str:
    path = TEMPLATES_DIR / filename
    if not path.exists():
        print(f"Error: Prompt file not found: {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def collect_job_files(output_dir: Path) -> list[Path]:
    """Find all .md files under output_dir, skipping summaries subdirectories."""
    files = []
    for md_file in sorted(output_dir.rglob("*.md")):
        if "summaries" in md_file.parts:
            continue
        files.append(md_file)
    return files


def job_summary_dir(md_file: Path, output_dir: Path, model: str) -> Path:
    """Return the per-job subdirectory where all category outputs are saved."""
    model_safe = model.replace("/", "-").replace(":", "-")
    safe_name = re.sub(r"[^\w\-]", "_", md_file.stem)
    return output_dir / model_safe / "summaries" / safe_name


def is_complete(md_file: Path, output_dir: Path, model: str) -> bool:
    """Return True if all category + overall outputs already exist for this job."""
    d = job_summary_dir(md_file, output_dir, model)
    return all((d / f"{cat}.json").exists() for cat in CATEGORIES + ["overall"])


def extract_job_title(md_file: Path) -> str:
    stem = md_file.stem
    title = re.sub(r"_\d{8}_\d{6}$", "", stem)
    return title.replace("_", " ")


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


async def call_gemini(client, prompt: str) -> str:
    """Run a single Gemini request in a thread executor using the hardcoded summarization model."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=f"models/{SUMMARIZE_MODEL}",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
        ),
    )
    return response.text


async def evaluate_category(client, category: str, report: str) -> dict:
    """Call Gemini for a single category and return {rating, reason}."""
    template = load_prompt(f"ranking_{category}.txt")
    prompt = template.format(report=report)
    raw = await call_gemini(client, prompt)
    return parse_json_response(raw)


async def evaluate_overall(client, job_title: str, ratings: dict) -> dict:
    """Call Gemini for the overall rating given the five assembled category ratings."""
    template = load_prompt("ranking_overall.txt")
    prompt = template.format(
        job_title=job_title,
        wlb_rating=ratings["wlb"]["rating"],
        wlb_reason=ratings["wlb"]["reason"],
        stress_rating=ratings["stress"]["rating"],
        stress_reason=ratings["stress"]["reason"],
        burnout_rating=ratings["burnout"]["rating"],
        burnout_reason=ratings["burnout"]["reason"],
        morning_hours_rating=ratings["morning_hours"]["rating"],
        morning_hours_reason=ratings["morning_hours"]["reason"],
        thinking_style_rating=ratings["thinking_style"]["rating"],
        thinking_style_reason=ratings["thinking_style"]["reason"],
    )
    raw = await call_gemini(client, prompt)
    return parse_json_response(raw)


async def summarize_file(
    client,
    md_file: Path,
    output_dir: Path,
    input_model: str,
    job_number: int,
    total: int,
    force: bool = False,
) -> dict:
    """Evaluate all categories in parallel, then get overall rating, save JSON per call."""
    label = f"[{job_number}/{total}]"
    job_title = extract_job_title(md_file)
    summary_dir = job_summary_dir(md_file, output_dir, input_model)

    try:
        report_text = md_file.read_text(encoding="utf-8")
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Only run categories whose output file is missing (or --force)
        cats_to_run = [cat for cat in CATEGORIES if force or not (summary_dir / f"{cat}.json").exists()]
        if cats_to_run:
            category_results = await asyncio.gather(
                *[evaluate_category(client, cat, report_text) for cat in cats_to_run]
            )
            for cat, result in zip(cats_to_run, category_results):
                (summary_dir / f"{cat}.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )

        # Load all category results from disk
        ratings = {
            cat: json.loads((summary_dir / f"{cat}.json").read_text(encoding="utf-8"))
            for cat in CATEGORIES
        }

        # Overall call: re-run if forced, if new categories were just computed, or if missing
        overall_path = summary_dir / "overall.json"
        if force or cats_to_run or not overall_path.exists():
            overall = await evaluate_overall(client, job_title, ratings)
            overall_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            overall = json.loads(overall_path.read_text(encoding="utf-8"))

        overall_label = overall["overall_rating"]
        print(f"{label} ✓  {job_title}  [{overall_label}]  → {summary_dir}/")
        return {"job": job_title, "status": "success", "dir": str(summary_dir), "overall_rating": overall_label}

    except Exception as exc:
        print(f"{label} ✗  {job_title}  — {exc}")
        return {"job": job_title, "status": "failed", "error": str(exc)}


async def run(client, files: list[Path], output_dir: Path, input_model: str, concurrency: int, force: bool):
    """Process all files with a concurrency limit (each job uses 5 calls internally)."""
    sem = asyncio.Semaphore(concurrency)
    total = len(files)

    async def bounded(md_file, idx):
        async with sem:
            return await summarize_file(client, md_file, output_dir, input_model, idx, total, force)

    tasks = [bounded(f, i + 1) for i, f in enumerate(files)]
    return await asyncio.gather(*tasks, return_exceptions=True)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Rank job research files on WLB, stress, burnout, and morning hours using Gemini"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root outputs directory containing model subdirs (default: outputs)",
    )
    parser.add_argument(
        "--input-model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model subfolder to read job .md files from (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of jobs to process concurrently — each uses 6 Gemini calls (default: 5)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (default: GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single .md job file to summarize instead of scanning the whole output dir",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process and overwrite jobs that already have a saved summary",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: No API key found. Set GEMINI_API_KEY or use --api-key.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' does not exist.")
        sys.exit(1)

    input_model = args.input_model

    if args.file:
        single = Path(args.file)
        if not single.exists():
            print(f"Error: File '{single}' does not exist.")
            sys.exit(1)
        files = [single]
    else:
        input_model_safe = input_model.replace("/", "-").replace(":", "-")
        input_dir = output_dir / input_model_safe
        if not input_dir.exists():
            print(f"Error: Input model directory '{input_dir}' does not exist.")
            sys.exit(1)
        files = collect_job_files(input_dir)

    if not files:
        print("No .md job files found under", output_dir)
        sys.exit(0)

    if not args.force:
        skipped = [f for f in files if is_complete(f, output_dir, input_model)]
        files = [f for f in files if not is_complete(f, output_dir, input_model)]
        if skipped:
            print(f"Skipping {len(skipped)} already-summarized job(s) (use --force to overwrite)")

    if not files:
        print("Nothing to do — all jobs already summarized.")
        sys.exit(0)

    print(f"Found {len(files)} job file(s) to process")
    print(f"Input model folder: {input_model}")
    print(f"Summarization model: {SUMMARIZE_MODEL} (thinking: high)")
    print(f"Approach: 5 parallel category calls + 1 overall call per job")
    print(f"Summaries → outputs/{input_model}/summaries/")
    print(f"Concurrency: {args.concurrency} job(s) at a time")
    print("=" * 70)

    client = genai.Client(api_key=api_key)

    start = datetime.now()
    results = asyncio.run(run(client, files, output_dir, input_model, args.concurrency, args.force))
    elapsed = (datetime.now() - start).total_seconds()

    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    failed     = [r for r in results if isinstance(r, dict) and r.get("status") == "failed"]

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Successful : {len(successful)}")
    print(f"  Failed     : {len(failed)}")
    print(f"  Duration   : {elapsed:.1f}s")

    if successful:
        order = ["Highly Recommended", "Recommended", "Neutral", "Not Recommended", "Avoid"]
        ranked = sorted(successful, key=lambda r: order.index(r["overall_rating"]) if r["overall_rating"] in order else 99)
        print("\nJobs by overall rating:")
        for r in ranked:
            print(f"  {r['overall_rating']:<22}  {r['job']}")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  - {r['job']}: {r.get('error', '?')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
