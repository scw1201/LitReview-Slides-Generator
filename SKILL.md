---
name: zotero-pdf-to-litreview-ppt
description: Generate a structured literature-review PPT and Markdown report from PDFs already attached to a specified local Zotero collection, without downloading new papers or using network calls. Use when the user asks to synthesize local Zotero PDFs into group-meeting style slides with cross-paper clustering, per-paper compressed analysis, evidence traceability, and fixed three-part deck structure.
---

# Zotero PDF To LitReview PPT

Generate a three-part literature-review deck from local Zotero PDFs only:

1. Overview slide (`文献整体结构梳理`): 2-5 clustered directions.
2. Per-paper slides: one compressed page per paper.
3. Final slide (`结构归纳与研究机会`): cross-paper synthesis and gaps.

## Constraints

- Read local PDFs only.
- Do not download new papers.
- Do not call network APIs.
- Do not modify Zotero data.

## Inputs

Primary mode (recommended): pass only Zotero collection name.

- Script auto-reads local Zotero DB: `~/Zotero/zotero.sqlite`
- Script auto-resolves PDF paths from: `~/Zotero/storage/`

Optional mode: pass a prebuilt manifest JSON.

- See `references/input_manifest_schema.md`.
- Required per item: `pdf_path`.
- Optional metadata: `title`, `authors`, `institution`, `venue`, `year`, `doi`.

## Run

Install dependencies:

```bash
python3 -m pip install -r scripts/requirements.txt
```

Generate outputs from Zotero collection directly:

```bash
python3 scripts/build_litreview.py \
  --collection "your_collection" \
  --max_papers 20 \
  --language zh \
  --include_images true \
  --output_dir /Users/la/Desktop/research_skills/
```

Optional: run with explicit manifest:

```bash
python3 scripts/build_litreview.py \
  --collection "your_collection" \
  --manifest /absolute/path/to/manifest.json \
  --language zh
```

## Outputs

- `/Users/la/Desktop/research_skills/review_<collection>.pptx`
- `/Users/la/Desktop/research_skills/review_<collection>.md`
- `/Users/la/Desktop/research_skills/review_<collection>.json`
- `/Users/la/Desktop/research_skills/review_<collection>.manifest.json` (only in auto-Zotero mode)
- `/Users/la/Desktop/research_skills/review_<collection>.status.json` (progress status)
- `/Users/la/Desktop/research_skills/review_<collection>.run.log` (run log)

JSON is the intermediate single source of truth for rendering Markdown and PPTX.

## Notes

- Clustering is deterministic (`random_state=42`, `k=2..5` with silhouette selection).
- If image extraction fails, slide image area stays blank with fallback text.
- If no processable PDF remains, emit an empty report instead of hard crash.

## Monitoring & GUI

CLI run now emits:

- `status.json`: stage, percentage, progress counters, message.
- `run.log`: timestamped runtime logs.

Optional GUI:

```bash
streamlit run scripts/gui.py
```

The GUI can start runs and display live status/log output from the same output directory.
