#!/usr/bin/env python3
"""
Batch process jobs from jobs.txt file using asyncio for concurrent research.
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from google import genai
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv


def load_jobs(filename: str = "jobs.txt"):
    """Load job names from the jobs file."""
    jobs_file = Path(filename)
    if not jobs_file.exists():
        print(f"Error: {filename} not found")
        sys.exit(1)

    jobs = []
    with open(jobs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(line)
    return jobs


def load_template(template_name: str = "job_prompt.j2"):
    """Load the Jinja template."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    return env.get_template(template_name)


def job_already_done(job_name: str, output_dir: str) -> bool:
    """Return True if a .md output file already exists for this job."""
    output_path = Path(output_dir) / "deep-research"
    safe_filename = job_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    return any(output_path.glob(f"{safe_filename}_*.md"))


async def research_job(client, job_name: str, template, output_dir: str, job_number: int, total_jobs: int, force: bool = False):
    """Research a single job asynchronously."""
    job_id = f"[{job_number}/{total_jobs}]"

    if not force and job_already_done(job_name, output_dir):
        print(f"{job_id} Skipping (already done): {job_name}")
        return {"job": job_name, "status": "skipped"}

    try:
        print(f"{job_id} Starting research: {job_name}")

        # Render the prompt
        prompt = template.render(job_name=job_name)

        # Start the deep research interaction
        interaction = client.interactions.create(
            input=prompt,
            agent="deep-research-pro-preview-12-2025",
            background=True
        )

        interaction_id = interaction.id
        print(f"{job_id} Research task started for {job_name} (ID: {interaction_id[:20]}...)")

        # Poll for completion
        while True:
            await asyncio.sleep(10)  # Poll every 10 seconds
            status = client.interactions.get(interaction_id)

            if status.status == "completed":
                # Get the final output
                if status.outputs and len(status.outputs) > 0:
                    response = status.outputs[-1].text

                    # Save to markdown (in model-specific subdirectory)
                    model_safe = "deep-research"
                    output_path = Path(output_dir) / model_safe
                    output_path.mkdir(parents=True, exist_ok=True)

                    safe_filename = job_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{safe_filename}_{timestamp}.md"
                    filepath = output_path / filename

                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# {job_name}\n\n")
                        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                        f.write("---\n\n")
                        f.write(response)

                    print(f"{job_id} ✓ Completed: {job_name} -> {filepath}")
                    return {"job": job_name, "status": "success", "file": str(filepath)}
                else:
                    raise Exception("No output received")

            elif status.status == "failed":
                error_msg = getattr(status, 'error', 'Unknown error')
                raise Exception(f"Research failed: {error_msg}")

    except Exception as e:
        print(f"{job_id} ✗ Failed: {job_name} - {e}")
        return {"job": job_name, "status": "failed", "error": str(e)}


async def process_batch(client, jobs, template, output_dir, concurrency=3, force=False):
    """Process jobs with controlled concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    total_jobs = len(jobs)

    async def bounded_research(job_name, job_number):
        async with semaphore:
            return await research_job(client, job_name, template, output_dir, job_number, total_jobs, force)

    tasks = [bounded_research(job, i+1) for i, job in enumerate(jobs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: No API key found.")
        print("Please create a .env file with: GEMINI_API_KEY=your-key")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Batch process jobs using Gemini Deep Research with async concurrency"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting line number (1-indexed)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending line number (inclusive)"
    )
    parser.add_argument(
        "--jobs-file",
        type=str,
        default="jobs.txt",
        help="Path to jobs file (default: jobs.txt)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="job_prompt.j2",
        help="Template to use (default: job_prompt.j2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent research tasks (default: 3)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process and overwrite jobs that already have a saved output"
    )

    args = parser.parse_args()

    # Setup
    client = genai.Client(api_key=api_key)
    template = load_template(args.template)

    # Load jobs
    all_jobs = load_jobs(args.jobs_file)
    print(f"Loaded {len(all_jobs)} jobs from {args.jobs_file}")

    # Select range
    start_idx = args.start - 1
    end_idx = args.end if args.end else len(all_jobs)

    if start_idx < 0 or start_idx >= len(all_jobs):
        print(f"Error: Start index {args.start} is out of range (1-{len(all_jobs)})")
        sys.exit(1)

    jobs_to_process = all_jobs[start_idx:end_idx]
    print(f"Processing jobs {args.start} to {min(end_idx, len(all_jobs))} ({len(jobs_to_process)} total)")
    print(f"Concurrency: {args.concurrency} simultaneous research tasks\n")
    print("=" * 80)

    # Process all jobs concurrently
    results = asyncio.run(process_batch(
        client,
        jobs_to_process,
        template,
        args.output_dir,
        args.concurrency,
        args.force,
    ))

    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)

    successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    skipped   = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")
    failed    = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")

    print(f"Total processed: {len(jobs_to_process)}")
    print(f"Successful: {successful}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed jobs:")
        for r in results:
            if isinstance(r, dict) and r.get("status") == "failed":
                print(f"  - {r['job']}: {r.get('error', 'Unknown error')}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
