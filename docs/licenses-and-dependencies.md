# Harvey Local AI — Licenses and Dependencies Audit

This document provides a comprehensive legal and technical audit of all software components, machine learning models, runtime dependencies, and the source code license of the Harvey Local AI Sales Agent project.

---

## 1. Repository Source Code License

- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **File:** `LICENSE`
- **Copyright:** (C) 2026 Free Software Foundation, Inc. / Sellbird Authors
- **Legal Obligations:**
  - **Copyleft / Share-Alike:** Any modifications and derived software interacting over a network (e.g., via the FastAPI HTTP interface) must make their corresponding source code available under AGPL-3.0.
  - **Patents:** Express patent grant from contributors.
  - **Notice Preservation:** All original copyright notices, disclaimers, and warranty limitations must be preserved in source files.
  - **Commercial Use:** Commercial deployment is permitted provided the AGPL copyleft requirements are satisfied for network users.

---

## 2. Artificial Intelligence Models & Embeddings

| Component | Model Name / Checkpoint | Primary Author / Provider | Declared License | Commercial Use Allowed | Local Offline Operation |
|---|---|---|---|---|---|
| **Primary LLM** | `qwen3:4b` | Alibaba Cloud / Qwen Team | **Apache 2.0** | Yes | Yes (via local Ollama daemon) |
| **Semantic Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face / Sentence-Transformers | **Apache 2.0** | Yes | Yes (weights stored locally in `data/models/embeddings/`) |

### Key Model Licensing Highlights
- **No Proprietary Cloud Model Weights:** No models from OpenAI, Anthropic, or Google are bundled or loaded.
- **Apache 2.0 Freedom:** Both `qwen3:4b` and `all-MiniLM-L6-v2` grant perpetual, worldwide, non-exclusive, no-charge, royalty-free copyright and patent licenses for both commercial and research applications.

---

## 3. Python Runtime Dependencies

All Python dependencies installed in the virtual environment (`.venv`) were inspected directly via `importlib.metadata` for their declared licenses:

| Package | Version | Declared License | Category / Function |
|---|---|---|---|
| `fastapi` | 0.163.1 | **MIT** | Core REST & Web API framework |
| `uvicorn` | 0.54.0 | **BSD-3-Clause** | ASGI server implementation |
| `pydantic` | 2.14.3 | **MIT** | Data validation and schema definitions |
| `httpx` | 0.28.1 | **BSD-3-Clause** | Asynchronous HTTP client (loopback Ollama) |
| `starlette` | 1.10.0 | **BSD-3-Clause** | ASGI foundation for FastAPI |
| `sentence-transformers` | 6.3.0 | **Apache-2.0** | Local dense vector embeddings |
| `torch` | 2.15.0 | **Apache-2.0 / BSD / MIT** | PyTorch tensor compute engine |
| `numpy` | 2.5.5 | **BSD-3-Clause** | Vector and array operations |
| `playwright` | 1.65.2 | **Apache-2.0** | Browser automation & acceptance testing |
| `aiosqlite` | 0.22.1 | **MIT / Permissive** | Asynchronous SQLite driver |
| `beautifulsoup4` | 4.15.0 | **MIT** | Offline HTML document parsing |
| `pyyaml` | 6.0.3 | **MIT** | YAML configuration parsing |

### Dependency Compatibility Conclusion
All runtime dependencies use standard permissive open-source licenses (**MIT**, **BSD-3-Clause**, or **Apache-2.0**). None contain GPL/AGPL conflicts with the repository's overarching **AGPL-3.0** license.

---

## 4. Zero Paid Cloud Services Audit & Classification

A codebase audit across all modules distinguishes active dependencies from optional/disabled integrations:

1. **Active Local AI Inference Runtime:**
   - Exclusively communicates over loopback `http://127.0.0.1:11434`.
   - Hostname and IP address are validated via `validate_loopback_url()`; non-loopback addresses (including LAN IPs) raise an immediate `ValueError`.
   - Zero automated fallback to cloud LLM APIs.
2. **Disabled Legacy AI Integrations:**
   - `sellbird/claude_runtime.py` and `sellbird/codex.py` are inert.
   - Any attempt to configure `provider: claude` or `provider: codex` raises an explicit Pydantic `ValidationError` via `PROHIBITED_CLOUD_PROVIDERS`.
3. **Optional External Third-Party Integrations:**
   - *Cloudflare Browser Rendering:* Present in `trainer.py` as an optional module; by default, site extraction uses the built-in local `FallbackCrawler` (HTTP + BeautifulSoup) requiring no Cloudflare credentials or subscriptions.
   - *Email Verification Providers (Hunter.io, ZeroBounce):* Optional auxiliary lookups in `email_finder.py`; default operation uses standard local DNS MX lookups and direct SMTP handshake heuristics.
4. **Development & Testing Only:**
   - `pytest`, `pytest-asyncio`, `playwright`.
