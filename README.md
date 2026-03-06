# Delivery Dashboard Generator

A web app that generates a 9-panel analytics dashboard from task delivery data. Upload your files, click Generate, download the PNG.

![Dashboard Example](https://img.shields.io/badge/Python-3.9+-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-green)

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:5050](http://localhost:5050) in your browser.

## How It Works

1. **Upload** your data files through the web UI
2. **Click** Generate Dashboard
3. **Download** the resulting PNG

### Required Files

| File | Format | Description |
|------|--------|-------------|
| **Delivery** | `.jsonl` | Main task data with model outputs and rubric grades |
| **DAMM Tasks** | `.csv` | MIME type metadata per task (`TASK_ID`, `ATTACHMENTS_S3_MIME_TYPE`, `GOLDEN_MIME_TYPE` columns) |

### Optional Files

| File | Format | Description |
|------|--------|-------------|
| **Grader Results** | `.jsonl` | Grader comparison data (Human / GPT-5.2 / Gemini) for the agreement matrix |
| **OCR Classification** | `.json` | OCR vs non-OCR disagreement analysis |

If grader results are not provided, the agreement matrix panel will show as empty.

## Dashboard Panels

### Top Row
1. **Domain Distribution** — task breakdown by domain
2. **Input File Type Distribution** — XLSX, JSON, PDF, etc.
3. **Output File Type Distribution** — PDF, PPTX, DOCX, etc.
4. **Input Files per Task** — count distribution with average
5. **Output Files per Task** — count distribution with average

### Bottom Row
6. **Total Rubrics per Task** — histogram with average
7. **Critical Rubrics per Task** — histogram with average
8. **Weighted Grade** — box plot of model scores (GPT5 excluded)
9. **Grader Agreement Matrix** — Human × GPT-5.2 × Gemini heatmap

## CLI Usage

The original standalone script is also included:

```bash
python generate_full_dashboard.py
```

This expects the data files in the same directory with their default names.

## Tech Stack

- **Backend:** Flask + matplotlib + numpy
- **Frontend:** Vanilla HTML/CSS/JS (dark theme matching the dashboard)
- **Output:** 24×14 inch PNG at 150 DPI
