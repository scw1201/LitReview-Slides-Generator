---
name: zotero-pdf-to-litreview-ppt
description: Generate Chinese or English literature-review markdown and PPT from local Zotero collections and local PDF attachments, with a 3-stage pipeline (per-paper analyze, global synthesis, render). Use when users ask to build/refresh lit-review slides from Zotero collection names, inspect per-paper progress in GUI, adjust pipeline config, or edit analysis JSON before rendering.
---

# Zotero PDF to LitReview PPT

Use this skill to run the local pipeline and produce `review_<collection>.pptx`.

## Run GUI

```bash
streamlit run /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/gui.py
```

Use GUI tabs in this order:
1. `Pipeline`: run Step 1/2/3 manually.
2. `Config`: edit and save pipeline config.
3. `Edit Analysis`: inspect and edit per-paper analysis JSON.

## Run CLI (stage by stage)

Step 1, analyze each paper:

```bash
OPENAI_API_KEY=ollama python3 /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py \
  --mode analyze \
  --collection "museum-digital-human" \
  --language zh \
  --llm_mode off \
  --output_dir /Users/la/Desktop/research_skills/
```

Step 2, global synthesis:

```bash
OPENAI_API_KEY=ollama python3 /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py \
  --mode global \
  --collection "museum-digital-human" \
  --cluster_k 4 \
  --output_dir /Users/la/Desktop/research_skills/
```

Step 3, render markdown and ppt:

```bash
OPENAI_API_KEY=ollama python3 /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py \
  --mode render \
  --collection "museum-digital-human" \
  --language zh \
  --include_images true \
  --output_dir /Users/la/Desktop/research_skills/
```

## Codex-First Quality Mode

When local model quality is unstable, prefer this flow:
1. Run Step 1 with `--llm_mode off` to produce base `review_<collection>.analyze.json`.
2. Edit paper fields in GUI `Edit Analysis` (task, method, contributions, limitations, keywords).
3. Run Step 2 + Step 3.

This keeps extraction deterministic and lets Codex improve final wording before rendering.

## Key Files

- `/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py`
- `/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/gui.py`
- `/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/config/pipeline.json`
- `/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/config/section_map.default.json`
