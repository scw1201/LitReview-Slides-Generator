# LitReview Slides Generator

Generate literature-review outputs (`.json`, `.md`, `.pptx`) from local Zotero PDFs.

## What This Project Does

- Reads papers from a Zotero collection (local database + local PDF attachments).
- Extracts per-paper structured analysis:
  - task definition
  - core method
  - main contributions
  - limitations
  - metadata and optional figure image
- Builds global synthesis and renders review slides.

## Project Structure

- `scripts/build_litreview.py`: main CLI pipeline.
- `scripts/gui.py`: Streamlit GUI.
- `scripts/requirements.txt`: Python dependencies.
- `config/section_map.default.json`: section heading regex mapping.
- `references/input_manifest_schema.md`: manifest schema notes.

## Quick Start

1. Install dependencies:

```bash
python3 -m pip install -r /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/requirements.txt
```

2. Run GUI:

```bash
streamlit run /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/gui.py
```

3. Or run CLI directly:

```bash
OPENAI_API_KEY=ollama python3 /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py \
  --collection "museum-digital-human" \
  --language zh \
  --include_images true \
  --llm_mode openai_compatible \
  --llm_model qwen3 \
  --llm_base_url http://127.0.0.1:11434/v1 \
  --llm_timeout_sec 180 \
  --llm_max_input_chars 4000 \
  --llm_max_tokens 512 \
  --section_map_json /Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/config/section_map.default.json \
  --output_dir /Users/la/Desktop/research_skills/ \
  --verbose
```

## Notes

- This branch can be used as a Codex Skill package (`SKILL.md` + `agents/openai.yaml`).
- No external paper download is required; local Zotero PDFs are used.
