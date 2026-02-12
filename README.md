# LitReview Slides Generator

一个从 **Zotero 本地文献库** 自动生成文献综述材料的工具：
- 输出 `PPT`（组会风格）
- 输出 `Markdown`（便于手改）
- 输出结构化 `JSON`（便于复用和二次开发）

主打三件事：**可控流程、可追踪状态、可编辑结果** ✨

## 功能全览 🚀

- 本地优先：只读本地 Zotero DB + 本地 PDF 附件
- 三阶段流水线：`analyze` → `global` → `render`
- 支持中文/英文输出：`--language zh|en`
- 单篇分析字段：任务定义、核心方法、主要贡献、局限、关键词
- 全局分析字段：研究方向聚类、跨论文归纳、研究空缺
- PPT 自动排版（16:9），支持插图（Fig.1 优先）
- GUI 支持：配置、分步运行、进度监控、单篇分析编辑
- 可选 RAG（基于 `zotero-mcp`）增强第二阶段总结

## 项目结构 📁

- `scripts/build_litreview.py`：主 CLI 管线
- `scripts/gui.py`：Streamlit GUI
- `scripts/requirements.txt`：依赖
- `config/pipeline.json`：默认配置
- `config/section_map.default.json`：章节匹配规则
- `references/input_manifest_schema.md`：输入结构参考
- `SKILL.md`、`agents/openai.yaml`：Codex skill 打包相关

## 环境要求 🧩

- Python 3.10+
- 本地安装并可访问的 Zotero
- 可选：
  - Codex CLI（推荐，`--llm_mode codex_cli`）
  - OpenAI-compatible 接口（如 Ollama）
  - `zotero-mcp`（用于 RAG）

## 安装 🔧

```bash
python3 -m pip install -r scripts/requirements.txt
```

## 一条命令全流程（CLI）⚡

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

## 分阶段运行（推荐）🪜

### Step 1: 单篇分析（analyze）

```bash
python3 scripts/build_litreview.py \
  --collection "museum-digital-human" \
  --mode analyze \
  --llm_mode codex_cli \
  --llm_model gpt-5-mini \
  --output_dir outputs
```

### Step 2: 全局聚类与总结（global）

```bash
python3 scripts/build_litreview.py \
  --collection "museum-digital-human" \
  --mode global \
  --cluster_k 3 \
  --llm_mode codex_cli \
  --llm_model gpt-5-mini \
  --output_dir outputs
```

### Step 3: 生成 Markdown/PPT（render）

```bash
python3 scripts/build_litreview.py \
  --collection "museum-digital-human" \
  --mode render \
  --language zh \
  --include_images true \
  --output_dir outputs
```

## GUI 使用 🖥️

启动：

```bash
streamlit run scripts/gui.py
```

你可以在 GUI 里：
- 在 `Pipeline` 页按步骤点击运行 Step1/Step2/Step3
- 在 `Config` 页统一维护参数（不需要手改 JSON）
- 在 `Edit Analysis` 页预览/编辑单篇分析并保存
- 查看当前运行状态、日志和每篇处理进度

## 输出目录规则 📦

默认 `--session_layout folder`：

- `outputs/<collection>/review_<collection>.manifest.json`
- `outputs/<collection>/review_<collection>.analyze.json`
- `outputs/<collection>/review_<collection>.global.json`
- `outputs/<collection>/review_<collection>.json`
- `outputs/<collection>/review_<collection>.md`
- `outputs/<collection>/review_<collection>.pptx`
- `outputs/<collection>/review_<collection>.status.json`
- `outputs/<collection>/review_<collection>.run.log`
- `outputs/<collection>/review_<collection>.paper_status.jsonl`

## LLM 模式 🤖

### 1) Codex CLI（推荐）

```bash
--llm_mode codex_cli --llm_model gpt-5-mini
```

可选指定可执行文件：

```bash
--codex_bin /Applications/Codex.app/Contents/Resources/codex
```

### 2) OpenAI-compatible（如 Ollama）

```bash
--llm_mode openai_compatible \
--llm_base_url http://127.0.0.1:11434/v1
```

如接口要求 API Key：

```bash
export OPENAI_API_KEY=your_key
```

## 可选 RAG（仅作用于 Step 2）🧠

使用 `zotero-mcp` 的语义检索增强全局方向归纳和 research gap 质量。

一次性建库（示例）：

```bash
zotero-mcp setup --semantic-config-only
zotero-mcp update-db --fulltext --force-rebuild
zotero-mcp db-status
```

开启 RAG：

```bash
--rag_enabled true \
--rag_top_k 8
```

常用 RAG 参数：
- `--rag_home_dir`：语义 DB 运行目录（建议可写目录）
- `--rag_config_path`：zotero-mcp 配置路径
- `--rag_use_local true`：本地 Zotero API 模式
- `--rag_python_bin`：安装了 `zotero_mcp` 的 Python

说明：RAG 失败不会中断流程，会自动回退，并在 `global.json.rag.last_error` 记录原因。


## 开发与贡献 ❤️

语法检查：

```bash
python3 -m py_compile scripts/build_litreview.py scripts/gui.py
```

贡献指南见：`CONTRIBUTING.md`

## 开源注意事项 🔐

- 仓库已忽略输出、日志、缓存、临时文件（见 `.gitignore`）
- 请勿提交本地 Zotero 数据、PDF、API Key
- 避免在配置和文档里写死个人绝对路径

## License

MIT，见 `LICENSE`。
