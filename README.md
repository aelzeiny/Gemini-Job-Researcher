# Gemini Job Researcher

Vibe-coded Job research pipeline using Google Gemini.

For example, I used it to research ~200 healthcare jobs on 5 constraints important to me (WLB, Thinking Stytle, Stress, Burnout, and Weekday 9-to-5 consistency).

Researches all jobs in `./jobs.txt` based on your own criteria & custom prompts in `./templates`.

[**Output Summary is publicly available here in Google Sheet**](https://docs.google.com/spreadsheets/d/1xaB_Ed2LMR7JlVFTRo7sTGkJtCYfFGg7fwCkC8cGrRs/edit?usp=sharing)

## What it does

1. **Researches** each job in `jobs.txt` (169 healthcare roles) using Gemini and saves detailed `.md` reports under `outputs/`.
2. **Rates** each job across 5 categories (WLB, stress, burnout, morning hours, thinking style) and produces an overall recommendation (`Highly Recommended` → `Avoid`).

## Setup

```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env
```

## Usage

```bash
# Step 1: Generate job reports
python batch_fast.py

# Step 2: Rate and summarize
python summarize_jobs.py

# Export to CSV
python export_csv.py
```

Results land in `outputs/<model>/` (reports) and `outputs/<model>/summaries/` (ratings).
