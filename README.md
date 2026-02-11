# LitReview Slides Generator

Generate structured literature-review slides (`.pptx`) and companion reports (`.md`, `.json`) from a **local Zotero collection**.

This project is built for research-group style reporting:
- Stage 1: per-paper analysis
- Stage 2: cross-paper clustering + synthesis (optional RAG)
- Stage 3: final rendering to markdown and PowerPoint

## Highlights

- Local-first pipeline (reads Zotero local DB + local PDF attachments)
- 3-stage workflow: `analyze` / `global` / `render`
- Chinese or English output (`--language zh|en`)
- Optional figure extraction (Fig.1-first strategy)
- Optional RAG-enhanced global synthesis via `zotero-mcp`
- Streamlit GUI for config/run/inspection/editing

## Repository Layout

- `scripts/build_litreview.py`: main CLI pipeline
- `scripts/gui.py`: Streamlit GUI
- `scripts/requirements.txt`: Python dependencies
- `config/pipeline.json`: default runtime config
- `config/section_map.default.json`: section heading mapping rules
- `references/input_manifest_schema.md`: manifest reference
- `agents/openai.yaml`: skill packaging metadata
- `SKILL.md`: Codex skill entry

## Requirements

- Python 3.10+
- Local Zotero desktop library
- Optional:
  - Codex CLI (`--llm_mode codex_cli`)
  - OpenAI-compatible endpoint (e.g., Ollama)
  - `zotero-mcp` for RAG in Stage 2

## Install

```bash
python3 -m pip install -r scripts/requirements.txt
```

## Quick Start (CLI)

Run all 3 stages in one command:

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

## Stage-by-Stage Usage

### Stage 1: Analyze (per paper)

```bash
python3 scripts/build_litreview.py \
  --collection "museum-digital-human" \
  --mode analyze \
  --llm_mode codex_cli \
  --output_dir outputs
```

### Stage 2: Global (clustering + synthesis)

```bash
python3 scripts/build_litreview.py \
  --collection "museum-digital-human" \
  --mode global \
  --llm_mode codex_cli \
  --output_dir outputs
```

### Stage 3: Render (ppt/md/json)

```bash
python3 scripts/build_litreview.py \
  --collection "museum-digital-human" \
  --mode render \
  --output_dir outputs
```

## Output Structure

Default (`--session_layout folder`):

- `outputs/<collection>/review_<collection>.manifest.json`
- `outputs/<collection>/review_<collection>.analyze.json`
- `outputs/<collection>/review_<collection>.global.json`
- `outputs/<collection>/review_<collection>.json`
- `outputs/<collection>/review_<collection>.md`
- `outputs/<collection>/review_<collection>.pptx`
- `outputs/<collection>/review_<collection>.status.json`
- `outputs/<collection>/review_<collection>.run.log`
- `outputs/<collection>/review_<collection>.paper_status.jsonl`

## GUI

```bash
streamlit run scripts/gui.py
```

GUI supports:
- Stage-triggered execution (Analyze / Global / Render)
- Config editing (single config source)
- Current-run status + per-paper progress
- Per-paper analysis preview/edit + save

## LLM Backends

### `codex_cli` (recommended)

```bash
--llm_mode codex_cli --llm_model gpt-5-mini
```

Optional explicit binary path:

```bash
--codex_bin /Applications/Codex.app/Contents/Resources/codex
```

### `openai_compatible` (e.g., Ollama)

```bash
--llm_mode openai_compatible \
--llm_base_url http://127.0.0.1:11434/v1
```

And set API key variable if endpoint requires it:

```bash
export OPENAI_API_KEY=your_key
```

## Optional RAG (Stage 2 only)

This project can use `zotero-mcp` semantic retrieval to improve global synthesis quality.

Recommended setup (one-time):

```bash
zotero-mcp setup --semantic-config-only
zotero-mcp update-db --fulltext --force-rebuild
zotero-mcp db-status
```

Then enable RAG in config or CLI:

```bash
--rag_enabled true \
--rag_top_k 8
```

Useful RAG runtime options:
- `--rag_home_dir`: writable home/cache directory for semantic DB runtime
- `--rag_config_path`: custom `zotero-mcp` semantic config
- `--rag_use_local true`: use local Zotero API mode
- `--rag_python_bin`: Python interpreter with `zotero_mcp` installed

If RAG retrieval fails, pipeline falls back to non-RAG synthesis and records reason in `global.json.rag.last_error`.

## Open-source Notes

- Generated files, logs, cache, and local runtime data are excluded by `.gitignore`.
- Avoid absolute local paths in committed configs/docs.
- Do not commit private Zotero data, PDFs, or API keys.

## Development

Basic syntax check:

```bash
python3 -m py_compile scripts/build_litreview.py scripts/gui.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT. See [LICENSE](LICENSE).
