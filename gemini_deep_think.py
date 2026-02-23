#!/usr/bin/env python3
"""
Gemini Deep Research Script
Takes a job name, renders a Jinja template, and saves Gemini's deep research response to markdown.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from google import genai
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from dotenv import load_dotenv


def setup_gemini(api_key: str):
    """Configure Gemini API with the provided key."""
    return genai.Client(api_key=api_key)


def load_template(template_name: str = "job_prompt.j2"):
    """Load the Jinja template from the templates directory."""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)

    try:
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_name)
        return template
    except TemplateNotFound:
        print(f"Error: Template '{template_name}' not found in {template_dir}")
        sys.exit(1)


def generate_deep_research(client, prompt: str):
    """Send prompt to Gemini Deep Research and poll for results."""
    print("Starting Gemini Deep Research...")
    print("This may take several minutes as the AI conducts thorough research.\n")

    # Start the deep research interaction
    interaction = client.interactions.create(
        input=prompt,
        agent="deep-research-pro-preview-12-2025",
        background=True
    )

    interaction_id = interaction.id
    print(f"Research task started (ID: {interaction_id})")

    # Poll for completion
    dots = 0
    while True:
        status = client.interactions.get(interaction_id)

        if status.status == "completed":
            print("\n✓ Research completed!")
            # Get the final output
            if status.outputs and len(status.outputs) > 0:
                return status.outputs[-1].text
            else:
                raise Exception("No output received from Deep Research")

        elif status.status == "failed":
            error_msg = getattr(status, 'error', 'Unknown error')
            raise Exception(f"Research failed: {error_msg}")

        elif status.status == "in_progress":
            # Show progress indicator
            dots = (dots + 1) % 4
            print(f"\rResearching{'.' * dots}{' ' * (3 - dots)}", end='', flush=True)
            time.sleep(5)  # Poll every 5 seconds

        else:
            print(f"\rUnknown status: {status.status}")
            time.sleep(5)


def save_to_markdown(job_name: str, content: str, output_dir: str = "outputs", model: str = "deep-research"):
    """Save the response to a markdown file."""
    # Create model-specific subdirectory
    model_safe = model.replace("/", "-").replace(":", "-")
    output_path = Path(output_dir) / model_safe
    output_path.mkdir(parents=True, exist_ok=True)

    # Create safe filename from job name
    safe_filename = job_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_filename}_{timestamp}.md"

    filepath = output_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {job_name}\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("---\n\n")
        f.write(content)

    print(f"Response saved to: {filepath}")
    return filepath


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate deep research analysis for a job using Gemini Deep Research API"
    )
    parser.add_argument(
        "job_name",
        type=str,
        help="Name of the job to analyze"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="job_prompt.j2",
        help="Jinja template file name (default: job_prompt.j2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output markdown files (default: outputs)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (default: reads from .env file or GEMINI_API_KEY env variable)"
    )

    args = parser.parse_args()

    # Get API key (prioritize command line arg, then env variable from .env)
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: No API key provided.")
        print("Please either:")
        print("  1. Create a .env file with: GEMINI_API_KEY=your-key")
        print("  2. Set environment variable: export GEMINI_API_KEY=your-key")
        print("  3. Use --api-key argument")
        sys.exit(1)

    # Setup
    client = setup_gemini(api_key)
    template = load_template(args.template)

    # Render the prompt
    prompt = template.render(job_name=args.job_name)
    print(f"Job: {args.job_name}")
    print(f"Template: {args.template}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)

    # Generate deep research response
    try:
        response = generate_deep_research(client, prompt)

        # Save to markdown
        filepath = save_to_markdown(args.job_name, response, args.output_dir, "deep-research")
        print(f"\n{'=' * 60}")
        print("Research complete and saved!")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\nError during deep research: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
