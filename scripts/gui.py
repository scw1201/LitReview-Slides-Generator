#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st


SCRIPT = "/Users/la/Desktop/check_code/zotero-pdf-to-litreview-ppt/scripts/build_litreview.py"
DEFAULT_OUTPUT = "/Users/la/Desktop/research_skills/"


def status_path(collection: str, outdir: str) -> Path:
    return Path(outdir) / f"review_{collection}.status.json"


def log_path(collection: str, outdir: str) -> Path:
    return Path(outdir) / f"review_{collection}.run.log"


def paper_status_path(collection: str, outdir: str) -> Path:
    return Path(outdir) / f"review_{collection}.paper_status.jsonl"


def report_path(collection: str, outdir: str) -> Path:
    return Path(outdir) / f"review_{collection}.json"


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def run_cmd(args, env):
    # Do not pipe stdout/stderr without consuming; it can block long-running jobs.
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def clear_state_files(collection: str, outdir: str) -> None:
    targets = [
        Path(outdir) / f"review_{collection}.status.json",
        Path(outdir) / f"review_{collection}.run.log",
        Path(outdir) / f"review_{collection}.paper_status.jsonl",
    ]
    for p in targets:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def items_to_text(items):
    lines = []
    for x in items or []:
        if isinstance(x, dict):
            t = str(x.get("text", "")).strip()
        else:
            t = str(x).strip()
        if t:
            lines.append(t)
    return "\n".join(lines)


def text_to_items(text):
    lines = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        s = s.lstrip("-").strip()
        if s:
            lines.append({"text": s, "evidence": []})
    return lines


st.set_page_config(page_title="Zotero LitReview PPT", layout="wide")
st.title("Zotero LitReview PPT Generator")

col1, col2 = st.columns(2)
with col1:
    collection = st.text_input("Collection", value="museum-digital-human")
    language = st.selectbox("Language", ["zh", "en"], index=0)
    max_papers = st.number_input("Max Papers", min_value=1, max_value=200, value=20, step=1)
    include_images = st.checkbox("Include Images", value=True)
with col2:
    llm_enabled = st.checkbox("Enable Local LLM", value=True)
    llm_model = st.text_input("LLM Model", value="qwen3-vl:4b")
    llm_base_url = st.text_input("LLM Base URL", value="http://127.0.0.1:11434/v1")
    llm_timeout_sec = st.number_input("LLM Timeout (sec)", min_value=30, max_value=900, value=180, step=30)
    llm_max_input_chars = st.number_input("LLM Max Input Chars", min_value=4000, max_value=40000, value=12000, step=1000)
    output_dir = st.text_input("Output Dir", value=DEFAULT_OUTPUT)

run_key = "run_pid"
if run_key not in st.session_state:
    st.session_state[run_key] = None

tab_pipeline, tab_edit = st.tabs(["Pipeline", "Edit Analysis JSON"])

with tab_pipeline:
    if st.button("Run"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # keep GUI status tied to one active process only
        old_pid = st.session_state.get(run_key)
        if old_pid:
            try:
                os.kill(old_pid, 15)
            except Exception:
                pass
        clear_state_files(collection, output_dir)
        args = [
            "python3",
            SCRIPT,
            "--collection",
            collection,
            "--language",
            language,
            "--max_papers",
            str(max_papers),
            "--include_images",
            "true" if include_images else "false",
            "--output_dir",
            output_dir,
        ]
        env = os.environ.copy()
        if llm_enabled:
            args.extend(
                [
                    "--llm_mode",
                    "openai_compatible",
                    "--llm_model",
                    llm_model,
                    "--llm_base_url",
                    llm_base_url,
                    "--llm_timeout_sec",
                    str(llm_timeout_sec),
                    "--llm_max_input_chars",
                    str(llm_max_input_chars),
                ]
            )
            env["OPENAI_API_KEY"] = "ollama"
        proc = run_cmd(args, env)
        st.session_state[run_key] = proc.pid
        st.success(f"Started PID={proc.pid}")

    st.caption("Click Refresh to reload status/log.")
    if st.button("Refresh"):
        pass

    auto_refresh = st.checkbox("Auto Refresh", value=True)
    refresh_sec = st.selectbox("Refresh Interval (sec)", [2, 3, 5, 10], index=3)

    pid = st.session_state.get(run_key)
    is_running = False
    if pid:
        try:
            os.kill(pid, 0)
            is_running = True
            st.info(f"Runner PID={pid} (running)")
        except Exception:
            st.warning(f"Runner PID={pid} (not running)")

    sp = status_path(collection, output_dir)
    lp = log_path(collection, output_dir)
    pp = paper_status_path(collection, output_dir)

    st.subheader("Status")
    if sp.exists():
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
            st.progress(int(data.get("progress_pct", 0)))
            st.json(data)
        except Exception as e:
            st.error(f"status parse failed: {e}")
    else:
        st.info("No status file yet.")

    st.subheader("Log")
    log_window = st.slider("Log Window (last N lines)", min_value=10, max_value=1000, value=10, step=10)
    if lp.exists():
        log_lines = lp.read_text(encoding="utf-8").splitlines()
        tail_lines = log_lines[-log_window:]
        st.code("\n".join(tail_lines), language="text")
        st.caption(f"Showing {len(tail_lines)} / {len(log_lines)} log lines")
    else:
        st.info("No log file yet.")

    st.subheader("Per-Paper Progress & Results")
    paper_rows = read_jsonl(pp)
    if paper_rows:
        df = pd.DataFrame(paper_rows)
        cols = [c for c in ["timestamp", "index", "total", "status", "title", "year", "venue", "authors_n", "has_image", "llm_task_used", "llm_method_used", "llm_contrib_used", "llm_limits_used", "llm_error", "reason"] if c in df.columns]
        st.dataframe(df[cols], width="stretch", height=280)

        parsed_rows = [r for r in paper_rows if r.get("status") == "parsed"]
        if parsed_rows:
            options = {f"[{r.get('index')}/{r.get('total')}] {r.get('title','unknown')}": r for r in parsed_rows}
            selected = st.selectbox("Inspect one paper", list(options.keys()))
            r = options[selected]
            st.markdown("**Task**")
            st.write("\n".join(r.get("task_definition", [])) or "-")
            st.markdown("**Core Method**")
            st.write("\n".join(f"- {x}" for x in r.get("core_method", [])) or "-")
            st.markdown("**Contributions**")
            st.write("\n".join(f"- {x}" for x in r.get("main_contributions", [])) or "-")
            st.markdown("**Limitations**")
            st.write("\n".join(f"- {x}" for x in r.get("limitations", [])) or "-")
            if r.get("llm_error"):
                st.markdown("**LLM Error**")
                st.code(str(r.get("llm_error")), language="text")
            if r.get("llm_error_debug"):
                st.markdown("**LLM Debug Error**")
                st.code(str(r.get("llm_error_debug")), language="text")
            if r.get("llm_raw_preview"):
                st.markdown("**LLM Raw Preview**")
                st.code(str(r.get("llm_raw_preview")), language="text")
    else:
        st.info("No per-paper status yet.")

    if auto_refresh and is_running:
        st.caption(f"Auto refreshing every {refresh_sec}s while runner is active")
        st.markdown(
            f"<meta http-equiv='refresh' content='{refresh_sec}'>",
            unsafe_allow_html=True,
        )

    st.subheader("Outputs")
    for suffix in ["pptx", "md", "json", "manifest.json", "status.json", "run.log"]:
        p = Path(output_dir) / f"review_{collection}.{suffix}"
        if p.exists():
            st.write(str(p))

with tab_edit:
    st.subheader("Edit Per-Paper Analysis Data")
    st.caption("Edit fields in review_<collection>.json and save for PPT rendering.")
    rp = report_path(collection, output_dir)
    st.code(str(rp), language="text")
    if not rp.exists():
        st.info("Analysis JSON not found yet. Run pipeline first.")
    else:
        try:
            report = json.loads(rp.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Failed to read report JSON: {e}")
            report = None

        if isinstance(report, dict):
            papers = report.get("papers", [])
            if not papers:
                st.info("No papers in report JSON.")
            else:
                indices = list(range(len(papers)))
                chosen = st.selectbox(
                    "Choose Paper",
                    indices,
                    format_func=lambda i: f"[{i + 1}/{len(papers)}] {papers[i].get('title', 'unknown')}",
                )
                p = papers[chosen]

                col_a, col_b = st.columns(2)
                with col_a:
                    title = st.text_input("Title", value=str(p.get("title", "")))
                    authors_text = st.text_input("Authors (comma-separated)", value=", ".join(p.get("authors", [])))
                    venue = st.text_input("Venue", value=str(p.get("venue", "")))
                    year = st.text_input("Year", value=str(p.get("year", "")))
                with col_b:
                    institution = st.text_input("Institution", value=str(p.get("institution", "")))
                    doi = st.text_input("DOI", value=str(p.get("doi", "")))
                    open_src = st.text_input(
                        "Open Source (yes/no/unknown)",
                        value=str((p.get("open_source_status", {}) or {}).get("value", "unknown")),
                    )
                    keywords = st.text_area("Keywords (one per line)", value="\n".join(p.get("keywords", [])), height=110)

                task_text = st.text_area("Task Definition (one line per bullet)", value=items_to_text(p.get("task_definition", [])), height=90)
                method_text = st.text_area("Core Method (one line per bullet)", value=items_to_text(p.get("core_method", [])), height=150)
                contrib_text = st.text_area("Main Contributions (one line per bullet)", value=items_to_text(p.get("main_contributions", [])), height=170)
                limit_text = st.text_area("Limitations (one line per bullet)", value=items_to_text(p.get("limitations", [])), height=150)

                save_backup = st.checkbox("Create backup before save", value=True)
                if st.button("Save Paper Edits"):
                    p["title"] = title.strip()
                    p["authors"] = [x.strip() for x in authors_text.split(",") if x.strip()]
                    p["venue"] = venue.strip()
                    p["year"] = year.strip()
                    p["institution"] = institution.strip()
                    p["doi"] = doi.strip()
                    p["keywords"] = [x.strip() for x in keywords.splitlines() if x.strip()]
                    p["task_definition"] = text_to_items(task_text)[:1]
                    p["core_method"] = text_to_items(method_text)[:3]
                    p["main_contributions"] = text_to_items(contrib_text)[:4]
                    p["limitations"] = text_to_items(limit_text)[:3]
                    p["open_source_status"] = {"value": (open_src.strip() or "unknown"), "evidence": []}

                    report["papers"][chosen] = p
                    if save_backup:
                        bak = rp.with_suffix(rp.suffix + f".{time.strftime('%Y%m%d-%H%M%S')}.bak")
                        bak.write_text(json.dumps(json.loads(rp.read_text(encoding="utf-8")), ensure_ascii=False, indent=2), encoding="utf-8")
                        st.caption(f"Backup: {bak}")
                    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.success("Saved changes to report JSON.")
