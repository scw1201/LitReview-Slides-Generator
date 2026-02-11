# LitReview Slides Generator

Generate literature-review outputs (`.json`, `.md`, `.pptx`) from a local Zotero collection and local PDF attachments.

## Features

- Read papers from Zotero local database and local storage.
- Per-paper analysis:
  - task definition
  - core method
  - main contributions
  - limitations
  - keywords
- Global synthesis with clustering and research gaps.
- Render outputs to markdown and PowerPoint.
- 3-stage pipeline (`analyze` / `global` / `render`) with GUI controls.
- Session-aware output organization.

## Project Structure

- `scripts/build_litreview.py`: main CLI pipeline.
- `scripts/gui.py`: Streamlit GUI.
- `scripts/requirements.txt`: dependencies.
- `config/pipeline.json`: default runtime config.
- `config/section_map.default.json`: section heading patterns.
- `references/input_manifest_schema.md`: manifest schema notes.

## Setup

From repository root:

```bash
python3 -m pip install -r scripts/requirements.txt
```

## Run GUI

```bash
streamlit run scripts/gui.py
```

## Run CLI (all stages)

```bash
python3 scripts/build_litreview.py \
  --collection "museum-digital-human" \
  --mode all \
  --language zh \
  --llm_mode codex_cli \
  --llm_model gpt-5-mini \
  --session_layout folder \
  --output_dir outputs \
  --verbose
```

## Stage Modes

- `--mode analyze`: build `analyze.json` per paper.
- `--mode global`: build `global.json` from `analyze.json`.
- `--mode render`: render final `.json/.md/.pptx`.
- `--mode all`: run all three stages.

## Output Layout

Default behavior (`--session_layout folder`):

- `outputs/<collection>/review_<collection>.manifest.json`
- `outputs/<collection>/review_<collection>.analyze.json`
- `outputs/<collection>/review_<collection>.global.json`
- `outputs/<collection>/review_<collection>.json`
- `outputs/<collection>/review_<collection>.md`
- `outputs/<collection>/review_<collection>.pptx`
- `outputs/<collection>/review_<collection>.status.json`
- `outputs/<collection>/review_<collection>.run.log`
- `outputs/<collection>/review_<collection>.paper_status.jsonl`

If `--session_name` is set and differs from collection:

- `outputs/<collection>/<session_name>/...`

If `--session_layout filename`:

- `outputs/review_<collection>.<session_name>.*`

## LLM Backends

### Codex CLI (recommended)

```bash
--llm_mode codex_cli --llm_model gpt-5-mini
```

Optional explicit binary path:

```bash
--codex_bin /Applications/Codex.app/Contents/Resources/codex
```

### OpenAI-compatible (e.g. Ollama)

```bash
--llm_mode openai_compatible --llm_base_url http://127.0.0.1:11434/v1
```

Set key env var if required by your endpoint:

```bash
export OPENAI_API_KEY=your_key
```

## Notes

- No external paper download is required.
- Pipeline reads local Zotero DB and local PDFs only.
- This repository can also be used as a Codex Skill package (`SKILL.md`, `agents/openai.yaml`).
