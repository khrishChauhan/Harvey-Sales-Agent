# Harvey Local AI Engine — Architecture & Operations Guide

## 1. Architectural Overview

Harvey Local AI Sales Agent transforms Sellbird from a cloud-dependent system (previously tethered to external Claude Code and Codex CLI runtimes) into an **entirely local, zero-cloud autonomous sales engine**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Harvey Sales Agent Pipeline                      │
│     [Scout (Discovery)] ──> [Writer (Drafting)] ──> [Handler (Replies)]│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         Brain Orchestration Layer
                               (sellbird/brain.py)
                                    │
                                    ▼
                          Brain Transport Seam
                       (sellbird/ollama_runtime.py)
                                    │
                     Strict Loopback Address Validation
                      (127.0.0.1 / localhost:11434)
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
        HTTP JSON Protocol                         SQLite Telemetry
        (/api/chat, /api/tags)                 (local_inference_logs)
               │
               ▼
     Local Ollama Runtime
    Model: qwen3:4b (Apache 2.0)
     GPU / CPU Hybrid Execution
```

### Key Guarantees
- **Zero Cloud Leakage**: No calls to Anthropic, OpenAI, Gemini, or any paid/third-party LLM endpoints.
- **Strict Loopback Binding**: All inference requests must target `http://127.0.0.1:11434` or `http://localhost:11434`. Non-loopback addresses are rejected by `validate_loopback_url` with a `ValueError`.
- **No Cloud AI Fallback**: If Ollama is offline or the model is missing, execution pauses cleanly. The engine will never switch silently to an external service.
- **Structured JSON Mode**: Local calls demanding structured output pass `"format": "json"` to Ollama's `/api/chat` endpoint, ensuring deterministic schema generation.
- **Preserved Pipeline Interfaces**: All existing sales agents (`Scout`, `Writer`, `Handler`, `Sender`, `Analyst`) consume the standard `Brain.think()` and `Brain.think_json()` interfaces without breaking changes.

---

## 2. Hardware Detection & Resource Evaluation

Harvey features automated hardware profiling in `sellbird/hardware.py`:
- **RAM**: Queries total and available physical RAM via Win32 `GlobalMemoryStatusEx`.
- **CPU**: Detects processor architecture and logical core counts.
- **GPU**: Queries NVIDIA GPU model, total VRAM, and free VRAM using `nvidia-smi`.

### Target Machine Profile
- **OS**: Windows 11
- **Host RAM**: 16 GB to 24 GB RAM (~9.3 GB free)
- **Host GPU**: NVIDIA GeForce RTX 2050 (4,096 MB VRAM, ~3,962 MB free)
- **Host CPU**: 12 logical cores (Intel Core architecture)

### Model Sizing & VRAM Arithmetic
For `qwen3:4b` quantized at Q4_K_M:
- **Model Weights**: ~2.55 GB
- **KV Cache (2,048 tokens context)**: ~0.55 GB
- **CUDA Runtime & Compute Context**: ~0.30 GB
- **Total Footprint**: ~3.40 GB

With ~3.9 GB free VRAM, `qwen3:4b` fits comfortably into GPU memory with minimal system paging. On machines where free VRAM falls below 3.6 GB, the engine gracefully coordinates partial layer offloading between GPU VRAM and system RAM.

---

## 3. Model Selection & Licensing

- **Primary Model**: `qwen3:4b` (dense open-weights model).
- **License**: **Apache 2.0** (permissive open-source license permitting commercial use, modification, and local self-hosting without recurring royalties or subscription dependencies).
- **Alternative Lightweight Option**: `qwen2.5:1.5b` (Apache 2.0) for low-memory environments (<2 GB VRAM).

---

## 4. Installation & Configuration

### Step 1: Install Ollama
Download and install Ollama for Windows from [ollama.com](https://ollama.com). Ensure the Ollama tray service is active.

### Step 2: Pull the Approved Model
Open PowerShell or Command Prompt and run:
```bash
ollama pull qwen3:4b
```
*(Model download is performed manually by the operator; Harvey never downloads multi-gigabyte models silently at startup.)*

### Step 3: Configure `sellbird.yaml`
Ensure `sellbird.yaml` contains the `brain` configuration block:
```yaml
brain:
  provider: "ollama"
  model: "qwen3:4b"
  endpoint: "http://127.0.0.1:11434"
  context_length: 2048
  temperature: 0.2

usage:
  max_daily_calls: 200
  heartbeat_interval_minutes: 15
```

---

## 5. CLI Commands & Verification

### Verify Health & Readiness
```bash
sellbird status --check
```
Output:
```
Sellbird Pipeline Status
========================================
...
Brain:          Ollama (qwen3:4b) ready at http://127.0.0.1:11434
```

### Inspect Local Usage & Telemetry
```bash
sellbird usage
```
Displays local Ollama engine parameters, zero cloud quota usage, and the latest local inferences logged to SQLite.

### Interactive Setup
```bash
sellbird setup
```
The setup wizard tests connection to `http://127.0.0.1:11434/api/tags`, checks that `qwen3:4b` is present, and executes a dry-run local ping.

---

## 6. SQLite Telemetry Schema

Local inference telemetry is stored in `data/sellbird.db` under the `local_inference_logs` table (Migration v17):

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Autoincrementing log ID |
| `provider` | TEXT | Always `'ollama'` |
| `model` | TEXT | Configured model name (`'qwen3:4b'`) |
| `agent` | TEXT | Sub-agent making the call (`scout`, `writer`, etc.) |
| `task` | TEXT | Specific task identifier (`score`, `draft_email`, etc.) |
| `duration_ms` | INTEGER | Round-trip inference latency in milliseconds |
| `prompt_tokens` | INTEGER | Input token count reported by Ollama |
| `completion_tokens` | INTEGER | Output token count reported by Ollama |
| `status` | TEXT | `'success'` or `'error'` |
| `error_message` | TEXT | Error description on failure, or empty string |
| `created_at` | TEXT | UTC timestamp of inference |

> [!NOTE]
> For operator privacy and database compactness, full prompt texts are **not** persisted in the database.

---

## 7. Automated Test Suite

Run the dedicated test suites to verify local engine integrity:

```bash
# Run Ollama runtime tests
pytest tests/test_ollama_runtime.py -v

# Run hardware detection tests
pytest tests/test_hardware.py -v
```
