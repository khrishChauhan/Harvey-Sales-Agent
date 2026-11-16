# Harvey Local AI Sales Intelligence Engine — Architecture & Operations Guide

## 1. Architectural Overview

Phase 3 elevates Harvey Local AI from a multi-product sales platform into an **autonomous, 100% local sales intelligence engine**.

All machine learning inferences, semantic representations, and NLP classifications execute on local hardware via **Ollama (`qwen3:4b`)** and **Sentence Transformers (`all-MiniLM-L6-v2`)**. The system maintains zero outbound calls to third-party or paid AI APIs, guarantees evidence-based reasoning without factual hallucination, and persists all vector and research state within standard SQLite.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Harvey Local AI Sales Intelligence Engine                    │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────────┐          ┌──────────────────────┐          ┌───────────────────┐
│ Local Embeddings │          │   Prospect Research  │          │   Local AI Brain  │
│  & Similarity    │          │      & Evidence      │          │     (Ollama)      │
└────────┬─────────┘          └──────────┬───────────┘          └─────────┬─────────┘
         │                               │                                │
         │   384-dim dense vectors       │   Verified facts vs            │   qwen3:4b
         │   all-MiniLM-L6-v2            │   AI interpretations           │   Format: JSON
         ▼                               ▼                                ▼
┌──────────────────┐          ┌──────────────────────┐          ┌───────────────────┐
│ SQLite Vector    │          │   Buying-Signal      │          │  Problem Statement│
│ Persistence      │          │   Classifier         │          │  & Pitch Hook     │
│ (embeddings tbl) │          │   (6 categories)     │          │  (Evidence-based) │
└────────┬─────────┘          └──────────┬───────────┘          └─────────┬─────────┘
         │                               │                                │
         └───────────────────────────────┼────────────────────────────────┘
                                         ▼
                     ┌───────────────────────────────────────┐
                     │    Explainable Composite Scoring      │
                     │  - Heuristics: Sector, Role, Email,   │
                     │    Signals (100-pt base)              │
                     │  - Semantic ICP Boost (up to +15 pts) │
                     │  - Transparent Factor Breakdown       │
                     └───────────────────┬───────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
             ┌──────────────────────┐        ┌───────────────────────┐
             │ Personalized Outreach│        │  Reply Classification │
             │ (Writer with Problem │        │  (Handler with 8-way  │
             │  Statement Injection)│        │   Product-Aware Flow) │
             └──────────────────────┘        └───────────────────────┘
```

---

## 2. Local Semantic Embeddings (`sellbird/embeddings.py`)

### 2.1 Model Specification
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **License**: **Apache 2.0** (Fully compliant with commercial and local assignment guidelines)
- **Vector Dimension**: 384 dimensions (dense float array)
- **Quantization/Disk Size**: ~80 MB total disk footprint
- **VRAM / RAM Impact**: ~120 MB RAM in CPU mode, ~200 MB VRAM in GPU mode (negligible impact on NVIDIA RTX 2050 4 GB VRAM)

### 2.2 Offline-First Design
- **Zero Automatic Downloads at Startup**: Harvey strictly enforces `allow_download=False` during regular application bootstrap and test runs to guarantee zero unexpected external network calls.
- **Dedicated Setup Utility**: A dedicated setup routine `download_model(model_name)` / CLI command downloads the model once to `data/models/embeddings/all-MiniLM-L6-v2`.
- **Completely Offline Operation**: Once cached locally, model loading uses `local_files_only=True` via HuggingFace transformers, functioning 100% offline.
- **Deterministic Heuristic Fallback**: If the embedding model has not yet been downloaded, Harvey gracefully falls back to deterministic token-level Jaccard similarity with word n-gram matching, preventing system panics or stalled pipelines.

### 2.3 SQLite Vector Persistence
Rather than introducing heavy, external vector databases (such as Pinecone, Qdrant, or Milvus), Harvey persists dense float embeddings directly into SQLite via Migration v19:

```sql
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,       -- 'prospect', 'company', 'product', 'icp'
    entity_id TEXT NOT NULL,         -- foreign entity key
    product_id TEXT DEFAULT '',      -- isolation by product
    embedding TEXT NOT NULL,         -- JSON serialized List[float]
    model_name TEXT NOT NULL,        -- 'all-MiniLM-L6-v2'
    dimension INTEGER NOT NULL,      -- 384
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_product ON embeddings(product_id);
```

### 2.4 Cosine Similarity & Matching
Vector similarity is computed via normalized dot product:
$$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
Normalized to $[0.0, 1.0]$. The state manager provides `state.find_similar_entities(entity_type, query_embedding, product_id, top_k)` to retrieve nearest neighbors in SQLite with product-boundary isolation.

---

## 3. Prospect Research Engine (`sellbird/research.py`)

### 3.1 Strict Separation of Verified Facts vs. AI Interpretations
To prevent factual hallucination and unverified claims in B2B outreach, Harvey strictly enforces data lineage:
- **Verified Facts**: Statements extracted directly from public company websites, job postings, or stored SQLite observations. Each verified fact carries an explicit `source_url`, `timestamp`, and `category` (e.g. `hiring`, `technology`, `metadata`, `heading`).
- **AI Interpretations**: Analytical deductions synthesized by local Ollama inference, clearly labeled as hypotheses or working assumptions.

### 3.2 Research Dossier Structure
```python
@dataclass
class ResearchDossier:
    company_id: str
    domain: str
    product_id: str = ""
    verified_facts: List[Dict[str, Any]]
    ai_interpretations: List[Dict[str, Any]]
    hiring_roles: List[str]
    tech_stack: List[str]
    source_urls: List[str]
    research_date: str
    offline_cached: bool = False
```

### 3.3 Offline Gathering
When `offline_only=True` is passed (or when outbound web requests are unavailable), `gather_company_research` reads solely from verified observations stored in SQLite (`observations` table). Existing public scraping metadata (HTML title, meta descriptions, and headers) extracted via BeautifulSoup is persisted into SQLite table `research_dossiers` for future offline runs.

---

## 4. Buying-Signal Classification (`sellbird/signals_classifier.py`)

Harvey classifies raw research observations into 6 standardized B2B buying signals:

| Signal Category | Description | Typical Evidence |
| :--- | :--- | :--- |
| `HIRING_EXPANSION` | Active recruitment for engineering, compliance, or leadership roles. | Careers page postings, ATS listings (Lever, Greenhouse) |
| `COMPANY_GROWTH` | Geographic expansion, team scaling, or funding milestones. | Press releases, public registry updates, office openings |
| `PRODUCT_LAUNCH` | Rollout of new solutions, beta platforms, or service tiers. | Product update blogs, header announcements |
| `TECH_ADOPTION` | Implementing or migrating modern cloud or security infrastructure. | Tech stack tags, script headers, bare-metal / k8s roles |
| `OPERATIONAL_CHALLENGES`| Managing legacy tooling complexity, friction, or audit bottlenecks. | Job requirements demanding remediation or audit scaling |
| `REGULATORY_COMPLIANCE` | Upcoming SOC2, HIPAA, ISO27001, or GDPR certification cycles. | Security page notices, compliance auditor hiring |

### 4.1 Local Ollama Classification
The classifier sends the verified evidence to local Ollama via `Brain.think_json()` with JSON mode enabled. Every classified signal outputs:
- `signal_type`: One of the 6 standardized enum categories.
- `title`: Short descriptive title.
- `relevance`: Confidence score (0.0 to 1.0) calibrated against the active product's value proposition.
- `confidence`: Reliability score of the evidence.
- `evidence`: Quoted text snippet from the verified facts.
- `source_url`: Traceable provenance URL.
- `is_verified`: Boolean ensuring provenance.

### 4.2 Offline Fallback
If Ollama is temporarily offline, a deterministic regex and keyword rule-engine extracts signals directly from evidence text, guaranteeing continuous pipeline throughput without interruption.

---

## 5. Evidence-based Problem Statement Generation (`sellbird/problem_statement.py`)

Rather than drafting generic cold outreach ("*I hope this email finds you well*"), Harvey synthesizes an evidence-grounded problem statement before generating copy.

### 5.1 Problem Statement Schema
```json
{
  "business_challenge": "FinBank is managing complex manual access controls across growing Kubernetes clusters while preparing for SOC2.",
  "supporting_evidence": [
    "Hiring 3 Cloud Security Architects",
    "Upcoming SOC2 Type II examination"
  ],
  "product_relevance": "ShieldZero replaces manual access reviews with automated mTLS zero-trust gating.",
  "uncertainties": [
    "Existing VPN vendor contract expiry date"
  ],
  "summary": "Automating zero-trust container access ahead of upcoming SOC2 audits."
}
```

### 5.2 Anti-Hallucination Guardrails
- **Grounding Rule**: The LLM prompt explicitly instructs the local model: *“Ground your analysis strictly in verified facts. If evidence is minimal, do NOT invent problems — state what is uncertain.”*
- **Writer Note Generation**: Problem statements emit a condensed `writer_note()` hook injected directly into the Writer agent prompt:
  ```text
  Core Problem Angle: FinBank is managing complex manual access controls...
  Observed Evidence: Hiring 3 Cloud Security Architects | Upcoming SOC2 examination
  Why We Fit: ShieldZero replaces manual access reviews with automated mTLS gating.
  ```

---

## 6. Explainable Composite Prospect Scoring (`sellbird/scoring.py`)

Harvey uses an explainable 0–100 scoring engine combining deterministic business heuristics with semantic embedding similarity:

```
Total Score (0–100) = Sector Match (0–20)
                    + Decision-Maker Title Match (0–30)
                    + Email Deliverability (0–25)
                    + Verified Buying Signals (0–25)
                    + Semantic ICP Similarity Boost (0–15)
```

### 6.1 Scoring Breakdown
1. **Sector / Industry Match (up to 20 pts)**:
   - Match against product's `target_industries`: +20 pts.
   - Industry mismatch: 0 pts + explicit negative factor logged.
2. **Decision-Maker Title Match (up to 30 pts)**:
   - Exact title match in product's `job_titles`: +30 pts.
   - Senior executive title (non-primary ICP): +15 pts.
   - Non-ICP title: 0 pts + negative factor logged.
3. **Deliverability Status (up to 25 pts)**:
   - `verified`: +25 pts.
   - `risky`: +12 pts.
   - `guess`: +5 pts (-20 deduction from potential).
   - `invalid`: 0 pts.
4. **Verified Buying Signals (up to 25 pts)**:
   - Active signal matching product ICP: +25 pts.
   - General hiring without exact role match: +5 pts.
   - No active signals: 0 pts + negative factor logged.
5. **Semantic ICP Similarity Boost (up to 15 pts)**:
   - If cosine similarity $\ge 0.65$: $\text{boost} = \text{round}(\text{similarity} \times 15)$.
   - Adds up to +15 points for prospects whose profiles align with product context.

### 6.2 Zero-Assumption Policy
Points are never awarded for missing data. If an industry or title is absent, 0 points are assigned and an explanation is added to `negative_factors`.

---

## 7. Personalised Email Generation & Reply Classification

### 7.1 Writer Agent Integration (`sellbird/agents/writer.py`)
- First email drafting automatically incorporates `ProblemStatement.writer_note()`.
- Incorporates product-specific value propositions, pricing anchors, and objection responses from the resolved `ProductContext`.
- Retains `product_id` across draft batches, preventing cross-product copy contamination.

### 7.2 Handler Agent Reply Classification (`sellbird/agents/handler.py`)
Incoming replies are classified into 8 granular intents:

1. `interested`: Positive interest in learning more.
2. `meeting_requested`: Direct request for a call or calendar booking.
3. `more_info_requested`: Request for technical docs, one-pagers, or pricing details.
4. `objection`: Specific hesitation (price, competitor, timing).
5. `not_interested`: Explicit polite or firm declination.
6. `follow_up_later`: Request to reconnect in next quarter / future date.
7. `out_of_office`: Automated vacation or absence responder.
8. `unclear`: Ambiguous or short non-committal response.

**Product-Context Isolation**: The Handler prompt dynamically receives the target `Product` name, description, and primary `Offer`, ensuring that objection handling and follow-up replies never quote the wrong product's pricing or terms.

---

## 8. Complete 9-Stage Offline Workflow (`sellbird/offline_workflow.py`)

Harvey orchestrates all intelligence components through an end-to-end pipeline callable via Python or CLI:

```python
from sellbird.offline_workflow import run_offline_sales_intelligence

report = await run_offline_sales_intelligence(
    state=state,
    brain=brain,
    company_id="comp_123",
    sample_reply="Could you send over technical docs for offline deployments?",
)

assert report.is_offline_verified is True
```

### The 9 Execution Stages:
1. **Business Profile Generation**: Local AI synthesis of company profile and brand voice.
2. **Product-specific ICP Generation**: Definition of buyer personas, target titles, and pain points.
3. **Prospect Research**: Offline retrieval of SQLite observations and verified facts.
4. **Semantic ICP Matching**: Sentence Transformers 384-dimensional cosine similarity.
5. **Buying-Signal Classification**: Local Ollama categorization into 6 signal types.
6. **Problem Statement Synthesis**: Evidence-backed challenge and outreach hook formulation.
7. **Prospect Scoring**: 0–100 composite scoring with positive/negative factor audit trail.
8. **Personalised Email Generation**: Problem-statement-grounded cold outreach email.
9. **Reply Classification**: Product-aware 8-way intent classification.

---

## 9. Verification & Test Suite

All Phase 3 features are thoroughly tested in `tests/test_sales_intelligence.py` and run alongside existing Phase 1 & 2 test suites:

```bash
# Execute Phase 3 Intelligence Tests
.venv\Scripts\pytest tests/test_sales_intelligence.py -v

# Execute Complete Project Test Suite (47 Tests)
.venv\Scripts\pytest tests/test_sales_intelligence.py tests/test_multi_product.py tests/test_product_api.py tests/test_ollama_runtime.py tests/test_hardware.py -v
```

### Test Coverage Breakdown:
- `test_local_embedding_generation_and_cosine_similarity`: 384-dim vector generation, Apache 2.0 license, cosine semantics.
- `test_semantic_prospect_icp_matching`: Semantic alignment and role matching.
- `test_embedding_sqlite_persistence`: SQLite save, retrieve, and top-k cosine similarity search.
- `test_buying_signal_classification`: Local AI classification and graceful fallback.
- `test_evidence_based_problem_statement`: Grounded problem synthesis and writer note generation.
- `test_explainable_prospect_scoring`: 100-pt score breakdown, positive factors, and negative deductions.
- `test_prospect_research_offline_gathering`: Offline observation retrieval and dossier generation.
- `test_reply_classification_phase3`: Handler 8-way classification with product context.
- `test_complete_offline_sales_intelligence_workflow`: Full 9-stage end-to-end pipeline execution offline.
