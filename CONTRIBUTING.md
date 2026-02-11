# Contributing

## Development Setup

```bash
python3 -m pip install -r scripts/requirements.txt
```

## Basic Checks

```bash
python3 -m py_compile scripts/build_litreview.py scripts/gui.py
```

## Pull Request Guidelines

- Keep changes focused and small.
- Update `README.md` when behavior or CLI arguments change.
- Avoid committing generated outputs (`outputs/`, logs, cache files).
- Avoid hard-coded absolute local paths.

## Commit Style

Use clear, imperative commit messages, for example:

- `feat: add rag support for global synthesis`
- `fix: handle empty llm response in per-paper analysis`
- `docs: update gui usage and output layout`
