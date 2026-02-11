#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str((PROJECT_ROOT / "scripts" / "build_litreview.py").resolve())
DEFAULT_OUTPUT = str((PROJECT_ROOT / "outputs").resolve())
DEFAULT_CONFIG = str((PROJECT_ROOT / "config" / "pipeline.json").resolve())
DEFAULT_SECTION_MAP = str((PROJECT_ROOT / "config" / "section_map.default.json").resolve())


def sanitize_session_name(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("._-")


def resolve_session_paths(collection: str, outdir: str, session_name: str, session_layout: str):
    sess = sanitize_session_name(session_name)
    root = Path(outdir)
    collection_dir = sanitize_session_name(collection) or collection
    base = f"review_{collection}"
    if (session_layout or "folder") == "filename":
        stem = f"{base}.{sess or collection_dir}"
        return root, stem
    coll_root = root / collection_dir
    if not sess or sess == collection_dir:
        return coll_root, base
    return coll_root / sess, base


def status_path(collection: str, outdir: str, session_name: str, session_layout: str) -> Path:
    d, stem = resolve_session_paths(collection, outdir, session_name, session_layout)
    return d / f"{stem}.status.json"


def log_path(collection: str, outdir: str, session_name: str, session_layout: str) -> Path:
    d, stem = resolve_session_paths(collection, outdir, session_name, session_layout)
    return d / f"{stem}.run.log"


def paper_status_path(collection: str, outdir: str, session_name: str, session_layout: str) -> Path:
    d, stem = resolve_session_paths(collection, outdir, session_name, session_layout)
    return d / f"{stem}.paper_status.jsonl"


def analyze_json_path(collection: str, outdir: str, session_name: str, session_layout: str) -> Path:
    d, stem = resolve_session_paths(collection, outdir, session_name, session_layout)
    return d / f"{stem}.analyze.json"


def global_json_path(collection: str, outdir: str, session_name: str, session_layout: str) -> Path:
    d, stem = resolve_session_paths(collection, outdir, session_name, session_layout)
    return d / f"{stem}.global.json"


def report_json_path(collection: str, outdir: str, session_name: str, session_layout: str) -> Path:
    d, stem = resolve_session_paths(collection, outdir, session_name, session_layout)
    return d / f"{stem}.json"


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
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def detect_build_processes():
    try:
        out = subprocess.check_output(["pgrep", "-fl", "build_litreview.py"], text=True)
        rows = []
        for line in out.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(line)
        return rows
    except Exception:
        return []


def clear_runtime_state(collection: str, outdir: str, session_name: str, session_layout: str) -> None:
    targets = [
        status_path(collection, outdir, session_name, session_layout),
        log_path(collection, outdir, session_name, session_layout),
        paper_status_path(collection, outdir, session_name, session_layout),
    ]
    for p in targets:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def default_config() -> Dict[str, Any]:
    return {
        "collection": "museum-digital-human",
        "language": "zh",
        "max_papers": 20,
        "cluster_k": 0,
        "include_images": True,
        "output_dir": DEFAULT_OUTPUT,
        "session_name": "",
        "session_layout": "folder",
        "llm_mode": "codex_cli",
        "llm_model": "gpt-5-mini",
        "llm_base_url": "http://127.0.0.1:11434/v1",
        "codex_bin": "",
        "llm_timeout_sec": 180,
        "llm_max_input_chars": 4000,
        "llm_max_tokens": 512,
        "rag_enabled": False,
        "rag_top_k": 8,
        "rag_python_bin": "",
        "rag_config_path": "",
        "rag_use_local": True,
        "section_map_json": DEFAULT_SECTION_MAP,
    }


def load_or_init_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        cfg = default_config()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return default_config()
        merged = default_config()
        merged.update(data)
        return merged
    except Exception:
        return default_config()


def is_pid_running(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


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


def render_paper_timeline(rows, limit=30):
    if not rows:
        st.info("No per-paper status yet for current run.")
        return
    # Show user-meaningful outcomes only.
    filtered = [r for r in rows if r.get("status") in {"parsed", "failed"}]
    if not filtered:
        st.info("No parsed/failed entries yet.")
        return
    recent = filtered[-limit:]
    for r in reversed(recent):
        idx = r.get("index", "-")
        total = r.get("total", "-")
        title = str(r.get("title", "unknown"))
        status = str(r.get("status", "unknown"))
        ts = str(r.get("timestamp", ""))
        badge = "✅" if status == "parsed" else ("❌" if status == "failed" else "⏳")
        with st.expander(f"{badge} [{idx}/{total}] {title}  ({status}, {ts})", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.caption(f"Year: {r.get('year', '-')}")
            c2.caption(f"Venue: {r.get('venue', '-')}")
            c3.caption(f"LLM Error: {r.get('llm_error', '-') or '-'}")
            c4.caption(f"Has Image: {r.get('has_image', '-')}")
            if r.get("task_definition"):
                st.markdown("**Task**")
                st.write("\n".join(r.get("task_definition", [])))
            if r.get("core_method"):
                st.markdown("**Method**")
                st.write("\n".join(f"- {x}" for x in r.get("core_method", [])))
            if r.get("main_contributions"):
                st.markdown("**Contributions**")
                st.write("\n".join(f"- {x}" for x in r.get("main_contributions", [])))
            if r.get("limitations"):
                st.markdown("**Limitations**")
                st.write("\n".join(f"- {x}" for x in r.get("limitations", [])))


st.set_page_config(page_title="LitReview Slides Generator", layout="wide")
st.title("LitReview Slides Generator")

if "config_path" not in st.session_state:
    st.session_state["config_path"] = DEFAULT_CONFIG
if "cfg" not in st.session_state:
    st.session_state["cfg"] = load_or_init_config(Path(st.session_state["config_path"]))

config_path = st.session_state["config_path"]
cfg = st.session_state["cfg"]
collection = str(cfg.get("collection", "museum-digital-human"))
output_dir = str(cfg.get("output_dir", DEFAULT_OUTPUT))
cfg_session_name = str(cfg.get("session_name", "")).strip()
cfg_session_layout = str(cfg.get("session_layout", "folder"))
if cfg_session_layout not in {"folder", "filename"}:
    cfg_session_layout = "folder"

if "active_session_name" not in st.session_state:
    st.session_state["active_session_name"] = cfg_session_name or (sanitize_session_name(collection) or collection)

if "run_pids" not in st.session_state:
    st.session_state["run_pids"] = {"analyze": None, "global": None, "render": None, "all": None}
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = False
if "refresh_sec" not in st.session_state:
    st.session_state["refresh_sec"] = 10
# one-time migration for old sessions that defaulted to auto refresh ON
if "auto_refresh_migrated" not in st.session_state:
    st.session_state["auto_refresh"] = False
    st.session_state["auto_refresh_migrated"] = True


def launch_mode(mode: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    session_layout = str(cfg.get("session_layout", "folder"))
    if session_layout not in {"folder", "filename"}:
        session_layout = "folder"
    collection_session = sanitize_session_name(collection) or collection
    configured = sanitize_session_name(str(cfg.get("session_name", "")))
    active = sanitize_session_name(str(st.session_state.get("active_session_name", "")))
    session_name = active or configured or collection_session
    if mode in {"analyze", "all"} and not session_name:
        session_name = collection_session
        st.session_state["active_session_name"] = session_name
    if mode in {"global", "render"} and not session_name:
        st.error("请先运行 Step 1，或在 Config 中设置 Session Name。")
        return
    # stop all old jobs first
    for m, pid in st.session_state["run_pids"].items():
        if is_pid_running(pid):
            try:
                os.kill(pid, 15)
            except Exception:
                pass
            st.session_state["run_pids"][m] = None
    if mode == "analyze":
        clear_runtime_state(collection, output_dir, session_name, session_layout)

    args = [
        "python3",
        SCRIPT,
        "--mode",
        mode,
        "--collection",
        collection,
        "--config_json",
        config_path,
        "--session_name",
        session_name,
        "--session_layout",
        session_layout,
    ]
    env = os.environ.copy()
    if str(cfg.get("llm_mode", "off")) == "openai_compatible":
        env["OPENAI_API_KEY"] = "ollama"
    proc = run_cmd(args, env)
    st.session_state["run_pids"][mode] = proc.pid
    st.success(f"Started {mode} PID={proc.pid} session={session_name}")


tab_pipeline, tab_config, tab_edit = st.tabs(["Pipeline", "Config", "Edit Analysis"])

with tab_pipeline:
    st.markdown("### Step 1 - Analyze Papers")
    st.caption("读取并分析每篇文献，输出 analyze.json")
    if st.button("Start Step 1", key="start_step1"):
        launch_mode("analyze")

    st.markdown("### Step 2 - Global Synthesis")
    st.caption("基于 analyze.json 做聚类方向分析与全局总结，输出 global.json")
    if st.button("Start Step 2", key="start_step2"):
        launch_mode("global")

    st.markdown("### Step 3 - Render PPT")
    st.caption("基于 analyze.json + global.json 生成 markdown/ppt/json")
    if st.button("Start Step 3", key="start_step3"):
        launch_mode("render")

    st.markdown("### Optional - Run Full Pipeline")
    if st.button("Start All Steps", key="start_all_steps"):
        launch_mode("all")

    auto_refresh = bool(st.session_state["auto_refresh"])
    refresh_sec = int(st.session_state["refresh_sec"])
    session_layout = str(cfg.get("session_layout", "folder"))
    if session_layout not in {"folder", "filename"}:
        session_layout = "folder"
    configured_session = sanitize_session_name(str(cfg.get("session_name", "")))
    active_session = sanitize_session_name(str(st.session_state.get("active_session_name", ""))) or configured_session

    running = {m: p for m, p in st.session_state["run_pids"].items() if is_pid_running(p)}
    external = detect_build_processes()
    active = bool(running or external)
    if running:
        st.info("Running: " + ", ".join([f"{m}(PID={p})" for m, p in running.items()]))
    elif external:
        st.warning("Detected running build process outside current GUI session.")
        st.code("\n".join(external), language="text")
    else:
        st.caption("No active process.")

    sp = status_path(collection, output_dir, active_session, session_layout)
    lp = log_path(collection, output_dir, active_session, session_layout)
    pp = paper_status_path(collection, output_dir, active_session, session_layout)

    st.subheader("Current Run Status")
    if not active:
        st.info("Idle. No active pipeline process.")
    else:
        status_data = {}
        if sp.exists():
            try:
                status_data = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                status_data = {}
        progress_pct = int(status_data.get("progress_pct", 0)) if status_data else 0
        st.progress(progress_pct)
        stage = str(status_data.get("stage", "-"))
        msg = str(status_data.get("message", "-"))

        paper_rows = read_jsonl(pp)
        parsed_rows = [r for r in paper_rows if r.get("status") == "parsed"]
        failed_rows = [r for r in paper_rows if r.get("status") == "failed"]
        current_row = paper_rows[-1] if paper_rows else {}
        cur_title = str(current_row.get("title", "-"))
        cur_index = current_row.get("index", "-")
        cur_total = current_row.get("total", "-")

        def stage_to_step(stage_name: str) -> str:
            s = (stage_name or "").strip().lower()
            if s in {"init", "manifest", "paper_analysis"}:
                return "1"
            if s in {"synthesis", "global"}:
                return "2"
            if s in {"render"}:
                return "3"
            if s == "done":
                if report_json_path(collection, output_dir, active_session, session_layout).exists():
                    return "3"
                if global_json_path(collection, output_dir, active_session, session_layout).exists():
                    return "2"
                return "1"
            return "-"

        m1, m2, m3, m4, m5 = st.columns(5)
        stage_label = stage_to_step(stage)
        if stage and stage != "-":
            stage_label = f"{stage_to_step(stage)} ({stage})"
        m1.metric("Stage", stage_label)
        m2.metric("Progress", f"{progress_pct}%")
        m3.metric("Parsed", str(len(parsed_rows)))
        m4.metric("Failed", str(len(failed_rows)))
        m5.metric("Current", f"{cur_index}/{cur_total}")
        st.caption(f"Message: {msg}")
        st.caption(f"Paper: {cur_title}")
        st.caption(f"Session: {active_session or '(none)'}")

        render_paper_timeline(paper_rows, limit=30)

        log_window = st.slider("Recent Log Lines", min_value=10, max_value=1000, value=80, step=10)
        if lp.exists():
            log_lines = lp.read_text(encoding="utf-8").splitlines()
            tail_lines = log_lines[-log_window:]
            st.code("\n".join(tail_lines), language="text")
        else:
            st.info("No log file yet.")

    st.subheader("Outputs")
    out_dir, out_stem = resolve_session_paths(collection, output_dir, active_session, session_layout)
    for p in [
        analyze_json_path(collection, output_dir, active_session, session_layout),
        global_json_path(collection, output_dir, active_session, session_layout),
        report_json_path(collection, output_dir, active_session, session_layout),
        out_dir / f"{out_stem}.md",
        out_dir / f"{out_stem}.pptx",
    ]:
        if p.exists():
            st.write(str(p))

    if auto_refresh and active:
        st.caption(f"Auto refreshing every {refresh_sec}s while process is active")
        st.markdown(f"<meta http-equiv='refresh' content='{refresh_sec}'>", unsafe_allow_html=True)

with tab_config:
    st.subheader("Pipeline Config")
    st.caption("Maintain pipeline config via form fields. No raw JSON editing needed.")
    cfg_file_col1, cfg_file_col2 = st.columns([4, 1])
    with cfg_file_col1:
        cfg_path_input = st.text_input("Config File", value=st.session_state["config_path"])
    with cfg_file_col2:
        if st.button("Load Config"):
            try:
                st.session_state["config_path"] = cfg_path_input
                st.session_state["cfg"] = load_or_init_config(Path(cfg_path_input))
                cfg_loaded = st.session_state["cfg"]
                coll_loaded = sanitize_session_name(str(cfg_loaded.get("collection", ""))) or str(cfg_loaded.get("collection", ""))
                sess_loaded = sanitize_session_name(str(cfg_loaded.get("session_name", "")))
                st.session_state["active_session_name"] = sess_loaded or coll_loaded
                st.success("Config loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Load failed: {e}")

    c1, c2 = st.columns(2)
    with c1:
        cfg_collection = st.text_input("Collection", value=str(cfg.get("collection", "museum-digital-human")))
        cfg_language = st.selectbox("Language", ["zh", "en"], index=0 if str(cfg.get("language", "zh")) == "zh" else 1)
        cfg_max_papers = st.number_input(
            "Max Papers",
            min_value=1,
            max_value=200,
            value=int(cfg.get("max_papers", 20)),
            step=1,
        )
        cfg_cluster_k = st.number_input(
            "Cluster K (0 = auto)",
            min_value=0,
            max_value=20,
            value=int(cfg.get("cluster_k", 0)),
            step=1,
        )
        cfg_include_images = st.checkbox("Include Images", value=bool(cfg.get("include_images", True)))
        cfg_output_dir = st.text_input("Output Dir", value=str(cfg.get("output_dir", DEFAULT_OUTPUT)))
        cfg_session_layout = st.selectbox(
            "Session Layout",
            ["folder", "filename"],
            index=0 if str(cfg.get("session_layout", "folder")) == "folder" else 1,
        )
        cfg_session_name = st.text_input(
            "Session Name (empty = auto on Step 1)",
            value=str(cfg.get("session_name", "")),
        )
    with c2:
        llm_modes = ["off", "codex_cli", "openai_compatible"]
        cur_mode = str(cfg.get("llm_mode", "codex_cli"))
        cur_index = llm_modes.index(cur_mode) if cur_mode in llm_modes else 1
        cfg_llm_mode = st.selectbox("LLM Mode", llm_modes, index=cur_index)
        cfg_llm_model = st.text_input("LLM Model", value=str(cfg.get("llm_model", "gpt-5-mini")))
        cfg_llm_base_url = st.text_input("LLM Base URL", value=str(cfg.get("llm_base_url", "http://127.0.0.1:11434/v1")))
        cfg_codex_bin = st.text_input(
            "Codex Bin",
            value=str(cfg.get("codex_bin", "")),
        )
        cfg_llm_timeout_sec = st.number_input(
            "LLM Timeout (sec)",
            min_value=30,
            max_value=900,
            value=int(cfg.get("llm_timeout_sec", 180)),
            step=30,
        )
        cfg_llm_max_input_chars = st.number_input(
            "LLM Max Input Chars",
            min_value=1000,
            max_value=50000,
            value=int(cfg.get("llm_max_input_chars", 4000)),
            step=500,
        )
        cfg_llm_max_tokens = st.number_input(
            "LLM Max Tokens",
            min_value=32,
            max_value=4000,
            value=int(cfg.get("llm_max_tokens", 512)),
            step=32,
        )
        cfg_rag_enabled = st.checkbox("Enable RAG in Global Stage", value=bool(cfg.get("rag_enabled", False)))
        cfg_rag_top_k = st.number_input(
            "RAG Top K",
            min_value=1,
            max_value=50,
            value=int(cfg.get("rag_top_k", 8)),
            step=1,
        )
        cfg_rag_python_bin = st.text_input(
            "RAG Python Bin (zotero-mcp env)",
            value=str(cfg.get("rag_python_bin", "")),
        )
        cfg_rag_config_path = st.text_input(
            "RAG Config Path (optional)",
            value=str(cfg.get("rag_config_path", "")),
        )
        cfg_rag_use_local = st.checkbox("RAG Use Local Zotero (ZOTERO_LOCAL=true)", value=bool(cfg.get("rag_use_local", True)))

    cfg_section_map_json = st.text_input(
        "Section Map JSON",
        value=str(
            cfg.get(
                "section_map_json",
                DEFAULT_SECTION_MAP,
            )
        ),
    )
    save_backup = st.checkbox("Create backup before save", value=True)
    st.markdown("#### Runtime Refresh")
    st.checkbox(
        "Auto Refresh (Pipeline)",
        key="auto_refresh",
    )
    refresh_choices = [2, 3, 5, 10]
    cur_refresh = int(st.session_state.get("refresh_sec", 10))
    refresh_idx = refresh_choices.index(cur_refresh) if cur_refresh in refresh_choices else 3
    st.selectbox(
        "Refresh Interval (sec)",
        refresh_choices,
        index=refresh_idx,
        key="refresh_sec",
    )
    st.markdown("#### Process Control")
    if st.button("Stop All Running build_litreview.py"):
        try:
            subprocess.run(["pkill", "-f", "build_litreview.py"], check=False)
            for k in list(st.session_state["run_pids"].keys()):
                st.session_state["run_pids"][k] = None
            active = sanitize_session_name(str(st.session_state.get("active_session_name", ""))) or sanitize_session_name(
                str(cfg.get("session_name", ""))
            )
            layout = str(cfg.get("session_layout", "folder"))
            clear_runtime_state(collection, output_dir, active, layout)
            st.success("Stop signal sent.")
        except Exception as e:
            st.error(f"Stop failed: {e}")
    if st.button("Save Config"):
        try:
            new_cfg = {
                "collection": cfg_collection,
                "language": cfg_language,
                "max_papers": int(cfg_max_papers),
                "cluster_k": int(cfg_cluster_k),
                "include_images": bool(cfg_include_images),
                "output_dir": cfg_output_dir,
                "session_layout": cfg_session_layout,
                "session_name": sanitize_session_name(cfg_session_name),
                "llm_mode": cfg_llm_mode,
                "llm_model": cfg_llm_model,
                "llm_base_url": cfg_llm_base_url,
                "codex_bin": cfg_codex_bin.strip(),
                "llm_timeout_sec": int(cfg_llm_timeout_sec),
                "llm_max_input_chars": int(cfg_llm_max_input_chars),
                "llm_max_tokens": int(cfg_llm_max_tokens),
                "rag_enabled": bool(cfg_rag_enabled),
                "rag_top_k": int(cfg_rag_top_k),
                "rag_python_bin": cfg_rag_python_bin.strip(),
                "rag_config_path": cfg_rag_config_path.strip(),
                "rag_use_local": bool(cfg_rag_use_local),
                "section_map_json": cfg_section_map_json,
            }
            st.session_state["config_path"] = cfg_path_input
            p = Path(cfg_path_input)
            if save_backup and p.exists():
                bak = p.with_suffix(p.suffix + f".{time.strftime('%Y%m%d-%H%M%S')}.bak")
                bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
                st.caption(f"Backup: {bak}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state["cfg"] = new_cfg
            coll_new = sanitize_session_name(str(new_cfg.get("collection", ""))) or str(new_cfg.get("collection", ""))
            sess_new = sanitize_session_name(str(new_cfg.get("session_name", "")))
            st.session_state["active_session_name"] = sess_new or coll_new
            st.success("Config saved and applied.")
        except Exception as e:
            st.error(f"Save failed: {e}")

with tab_edit:
    st.subheader("Edit Per-Paper Analysis Data")
    session_layout = str(cfg.get("session_layout", "folder"))
    if session_layout not in {"folder", "filename"}:
        session_layout = "folder"
    active_session = sanitize_session_name(str(st.session_state.get("active_session_name", ""))) or sanitize_session_name(
        str(cfg.get("session_name", ""))
    )
    rp = analyze_json_path(collection, output_dir, active_session, session_layout)
    st.code(str(rp), language="text")
    running_now = {m: p for m, p in st.session_state["run_pids"].items() if is_pid_running(p)}
    external_now = detect_build_processes()
    active_now = bool(running_now or external_now)
    editable = (not active_now) and rp.exists()

    if not rp.exists():
        st.warning("Analyze JSON not found yet. Showing preview from paper_status.jsonl if available.")
        pp = paper_status_path(collection, output_dir, active_session, session_layout)
        rows = read_jsonl(pp)
        parsed_rows = [r for r in rows if r.get("status") == "parsed"]
        if parsed_rows:
            st.caption("Preview only (not editable yet)")
            render_paper_timeline(parsed_rows, limit=20)
        else:
            st.info("No parsed paper preview yet.")
    else:
        if active_now:
            st.info("Preview mode: pipeline is running, editing is locked to avoid dirty read/write.")
        else:
            st.success("Edit mode: pipeline idle, safe to modify analyze.json.")
        try:
            report = json.loads(rp.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Failed to read analyze JSON: {e}")
            report = None

        if isinstance(report, dict):
            papers = report.get("papers", [])
            if not papers:
                st.info("No papers in analyze JSON.")
            else:
                idxs = list(range(len(papers)))
                chosen = st.selectbox(
                    "Choose Paper",
                    idxs,
                    format_func=lambda i: f"[{i + 1}/{len(papers)}] {papers[i].get('title', 'unknown')}",
                )
                p = papers[chosen]

                col_a, col_b = st.columns(2)
                with col_a:
                    title = st.text_input("Title", value=str(p.get("title", "")), disabled=not editable)
                    authors_text = st.text_input("Authors (comma-separated)", value=", ".join(p.get("authors", [])), disabled=not editable)
                    venue = st.text_input("Venue", value=str(p.get("venue", "")), disabled=not editable)
                    year = st.text_input("Year", value=str(p.get("year", "")), disabled=not editable)
                with col_b:
                    institution = st.text_input("Institution", value=str(p.get("institution", "")), disabled=not editable)
                    doi = st.text_input("DOI", value=str(p.get("doi", "")), disabled=not editable)
                    open_src = st.text_input(
                        "Open Source (yes/no/unknown)",
                        value=str((p.get("open_source_status", {}) or {}).get("value", "unknown")),
                        disabled=not editable,
                    )
                    keywords = st.text_area("Keywords (one per line)", value="\n".join(p.get("keywords", [])), height=110, disabled=not editable)

                task_text = st.text_area("Task Definition (one line per bullet)", value=items_to_text(p.get("task_definition", [])), height=90, disabled=not editable)
                method_text = st.text_area("Core Method (one line per bullet)", value=items_to_text(p.get("core_method", [])), height=150, disabled=not editable)
                contrib_text = st.text_area("Main Contributions (one line per bullet)", value=items_to_text(p.get("main_contributions", [])), height=170, disabled=not editable)
                limit_text = st.text_area("Limitations (one line per bullet)", value=items_to_text(p.get("limitations", [])), height=150, disabled=not editable)

                save_backup = st.checkbox("Create backup before save", value=True, key="paper_backup")
                if st.button("Save Paper Edits", disabled=not editable):
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
                    if save_backup and rp.exists():
                        bak = rp.with_suffix(rp.suffix + f".{time.strftime('%Y%m%d-%H%M%S')}.bak")
                        bak.write_text(rp.read_text(encoding="utf-8"), encoding="utf-8")
                        st.caption(f"Backup: {bak}")
                    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.success("Saved changes to analyze JSON.")
