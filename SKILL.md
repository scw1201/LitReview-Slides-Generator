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
python3 /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py \
  --mode analyze \
  --collection "museum-digital-human" \
  --language zh \
  --llm_mode codex_cli \
  --llm_model gpt-5-mini \
  --output_dir /Users/la/Desktop/research_skills/
```

Step 2, global synthesis:

```bash
python3 /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py \
  --mode global \
  --collection "museum-digital-human" \
  --cluster_k 4 \
  --llm_mode codex_cli \
  --llm_model gpt-5-mini \
  --output_dir /Users/la/Desktop/research_skills/
```

Step 3, render markdown and ppt:

```bash
python3 /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py \
  --mode render \
  --collection "museum-digital-human" \
  --language zh \
  --include_images true \
  --llm_mode codex_cli \
  --llm_model gpt-5-mini \
  --output_dir /Users/la/Desktop/research_skills/
```

## Codex-First Quality Mode

Use `--llm_mode codex_cli` in all stages so content generation is done by Codex.

## Key Files

- `/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py`
- `/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/gui.py`
- `/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/config/pipeline.json`
- `/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/config/section_map.default.json`
