#!/usr/bin/env python3
"""
Fast Gemini job analysis using generative models (not Deep Research).
Completes in seconds instead of minutes.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from google import genai
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from dotenv import load_dotenv


def setup_gemini(api_key: str):
    """Configure Gemini API with the provided key."""
    return genai.Client(api_key=api_key)


def load_template(template_name: str = "job_prompt_focused.j2"):
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


def generate_response(client, prompt: str, model: str = "gemini-3-flash-preview"):
    """Send prompt to Gemini and get response quickly."""
    print("Generating analysis...")

    response = client.models.generate_content(
        model=f"models/{model}",
        contents=prompt
    )

    return response.text


def save_to_markdown(job_name: str, content: str, output_dir: str = "outputs", model: str = "gemini-3-flash-preview"):
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
        description="Generate fast job analysis using Gemini (not Deep Research)"
    )
    parser.add_argument(
        "job_name",
        type=str,
        help="Name of the job to analyze"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="job_prompt_focused.j2",
        help="Jinja template file name (default: job_prompt_focused.j2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output markdown files (default: outputs)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model to use (default: gemini-3-flash-preview)"
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
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)

    # Generate response
    try:
        response = generate_response(client, prompt, args.model)

        # Save to markdown
        filepath = save_to_markdown(args.job_name, response, args.output_dir, args.model)
        print(f"\n{'=' * 60}")
        print("Analysis complete!")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\nError generating response: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
