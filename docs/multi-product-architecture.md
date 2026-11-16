# Harvey Multi-Product Architecture

## Overview

Harvey was originally architected as a single-product autonomous sales engine where ICP definitions, value propositions, and objection playbooks were declared as global process-wide singletons. 

Phase 2 transforms Harvey into a **Multi-Product Sales Engine** that natively supports:
- **Multiple Business Profiles**: Multiple independent legal or operational entities within a single workspace.
- **Multiple Products per Business**: Distinct offerings with independent pricing, key features, benefits, unique selling points (USPs), objections handling, and call-to-actions.
- **Multiple ICPs per Product**: Granular Ideal Customer Profiles per product targeting different verticals, personas, and qualification requirements.
- **Strict Context Isolation**: Zero cross-product context leakage during AI reasoning tasks (prospect scoring, sequence drafting, reply handling).
- **Local AI Generation**: On-device synthesis of Business Profiles and ICPs via Ollama (`qwen3:4b`), strictly enforcing factual grounding without hallucination.
- **Seamless Backward Compatibility**: Single-product installations continue running unmodified through Migration v18 and automatic default-product backfilling.

---

## 1. Database Schema & Architecture

The relational schema is backed by SQLite and managed through versioned asynchronous migrations (`sellbird/state.py`).

### Relational Entity Model

```
 ┌──────────────────────┐
 │  Business Profiles   │
 │──────────────────────│
 │ id (TEXT PRIMARY KEY)│
 │ name                 │
 │ description          │
 │ website              │
 │ industry             │
 │ company_size         │
 │ business_location    │
 │ target_markets (JSON)│
 │ brand_voice          │
 └──────────┬───────────┘
            │ 1:N
            ▼
 ┌──────────────────────┐
 │       Products       │
 │──────────────────────│
 │ id (TEXT PRIMARY KEY)│
 │ business_id (FK)     │◀─── Deletion cascade safeguards
 │ name                 │
 │ description          │
 │ category             │
 │ pricing              │
 │ key_features (JSON)  │
 │ key_benefits (JSON)  │
 │ usps (JSON)          │
 │ target_industries(JSON)
 │ objections (JSON)    │
 │ offer (JSON)         │
 └──────────┬───────────┘
            │ 1:N
            ├─────────────────────────────────────────────────┐
            ▼                                                 ▼
 ┌──────────────────────┐                          ┌──────────────────────┐
 │         ICPs         │                          │ Existing Entities    │
 │──────────────────────│                          │ (product_id FK)      │
 │ id (TEXT PRIMARY KEY)│                          │──────────────────────│
 │ product_id (FK)      │                          │ • prospects          │
 │ name                 │                          │ • campaigns          │
 │ description          │                          │ • conversations      │
 │ target_industries    │                          │ • observations       │
 │ company_sizes        │                          │ • outbox             │
 │ target_geographies   │                          └──────────────────────┘
 │ job_titles           │
 │ buyer_personas       │
 │ pain_points          │
 │ buying_signals       │
 │ qualification_criteria
 └──────────────────────┘
```

### Table Definitions

#### `businesses`
| Column | Type | Description |
|---|---|---|
| `id` | TEXT PRIMARY KEY | Unique business identifier (12 hex chars or UUID) |
| `name` | TEXT NOT NULL | Entity name |
| `description` | TEXT NOT NULL | Detailed business description |
| `website` | TEXT NOT NULL | Corporate website URL |
| `industry` | TEXT NOT NULL | Primary operating industry |
| `company_size` | TEXT NOT NULL | Headcount range (e.g. `50-200`) |
| `business_location` | TEXT NOT NULL | HQ or regional jurisdiction |
| `target_markets` | TEXT NOT NULL | JSON array of target regions/sectors |
| `brand_voice` | TEXT NOT NULL | Tone of voice (e.g., authoritative, consultative) |
| `created_at` | TEXT NOT NULL | ISO 8601 UTC timestamp |
| `updated_at` | TEXT NOT NULL | ISO 8601 UTC timestamp |

#### `products`
| Column | Type | Description |
|---|---|---|
| `id` | TEXT PRIMARY KEY | Unique product identifier |
| `business_id` | TEXT NOT NULL | Foreign key referencing `businesses(id)` |
| `name` | TEXT NOT NULL | Product or service name |
| `description` | TEXT NOT NULL | Comprehensive overview |
| `category` | TEXT NOT NULL | Taxonomy category |
| `pricing` | TEXT NOT NULL | Pricing model or rate card |
| `key_features` | TEXT NOT NULL | JSON array of technical capabilities |
| `key_benefits` | TEXT NOT NULL | JSON array of business value outcomes |
| `unique_selling_points` | TEXT NOT NULL | JSON array of market differentiators |
| `target_industries` | TEXT NOT NULL | JSON array of ICP industries |
| `objection_responses` | TEXT NOT NULL | JSON object mapping objections to rebuttals |
| `offer` | TEXT NOT NULL | JSON object detailing pilot / introductory offer |
| `created_at` | TEXT NOT NULL | ISO 8601 UTC timestamp |
| `updated_at` | TEXT NOT NULL | ISO 8601 UTC timestamp |

#### `icps`
| Column | Type | Description |
|---|---|---|
| `id` | TEXT PRIMARY KEY | Unique ICP identifier |
| `product_id` | TEXT NOT NULL | Foreign key referencing `products(id)` |
| `name` | TEXT NOT NULL | Segment label (e.g., "Fintech CISOs") |
| `description` | TEXT NOT NULL | Scope and strategic context |
| `target_industries` | TEXT NOT NULL | JSON array of industries |
| `company_sizes` | TEXT NOT NULL | JSON array of company size bands |
| `target_geographies` | TEXT NOT NULL | JSON array of regional constraints |
| `job_titles` | TEXT NOT NULL | JSON array of prospect decision-maker titles |
| `buyer_personas` | TEXT NOT NULL | JSON array of psychological & role drivers |
| `pain_points` | TEXT NOT NULL | JSON array of acute problems solved |
| `buying_signals` | TEXT NOT NULL | JSON array of trigger events |
| `qualification_criteria` | TEXT NOT NULL | JSON array of hard inclusion criteria |
| `created_at` | TEXT NOT NULL | ISO 8601 UTC timestamp |
| `updated_at` | TEXT NOT NULL | ISO 8601 UTC timestamp |

---

## 2. Migration & Backward Compatibility Strategy

### Safe Migration v18
Migration v18 in `sellbird/state.py` guarantees idempotent, non-destructive execution:
1. Creates the new `businesses`, `products`, and `icps` tables along with indexes on foreign keys (`idx_products_business`, `idx_icps_product`).
2. Checks existing tables (`prospects`, `campaigns`, `conversations`, `observations`, `outbox`) for the `product_id` column using `PRAGMA table_info(...)`.
3. Safely adds `product_id TEXT NOT NULL DEFAULT ''` to existing tables only if not already present.
4. Creates indexes `idx_prospects_product_id` and `idx_campaigns_product_id` for fast query filtering.

### Automatic Backfill: `ensure_default_product()`
To prevent disruption for single-product deployments or existing databases:
- When legacy records have empty `product_id = ''`, `state.ensure_default_product()` reads the workspace's `sellbird.yaml` (product name, description, objections, offer).
- Creates a Default Business Profile and Default Product if none exist.
- Executes `UPDATE ... SET product_id = :def_id WHERE product_id = '' OR product_id IS NULL` across all prospects, campaigns, conversations, observations, and outbox rows.
- Existing single-product workflows continue functioning with zero configuration changes.

---

## 3. Product Context Isolation Engine

Cross-product data contamination (e.g., pitching Product B's pricing to Product A's prospect, or using Product A's objection handling on Product B) is strictly prevented via `ProductContext` in `sellbird/product_context.py`.

```
                  ┌───────────────────────┐
                  │ Target Entity / Batch │
                  │  (product_id tagged)  │
                  └──────────┬────────────┘
                             │
                             ▼
              ┌───────────────────────────────┐
              │   resolve_product_context()   │
              └──────────────┬────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
  [Database Record]                     [Fallback sellbird.yaml]
  • Product Name, Features, Offer       • Single-product fallback
  • Objection Responses                 • Preserves existing behavior
  • ICP Job Titles, Signals
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │ Isolated ProductContext Injected into Agent  │
  │ • Writer: sequence drafting, dynamic angles  │
  │ • Handler: objection rebuttals, call-to-acts │
  │ • Scout: qualification & title scoring       │
  └──────────────────────────────────────────────┘
```

### Agent Integration:
- **`sellbird/agents/writer.py`**:
  - Groups prospect batches strictly by `product_id`.
  - Injects isolated `ProductContext` into AI sequence generation and personalized 1-to-1 emails.
- **`sellbird/agents/handler.py`**:
  - Looks up the active conversation's `product_id`.
  - Injects the specific product's objection responses, pilot offer, and messaging into the reply generation prompt.
- **`sellbird/agents/scout.py`**:
  - Evaluates decision-maker titles and qualification signals using the product's ICP criteria rather than global defaults.

---

## 4. Local AI Profile & ICP Generation

Business and ICP generation are executed entirely via local Ollama inference (`sellbird/brain.py`) using structured JSON outputs:

### 1. Business Profile Generator (`sellbird/business_generator.py`)
- Takes raw business name, optional website, notes, and confirmed facts.
- Prompt enforces strict factual grounding: **never invents certifications, unverified customer numbers, or fabricated capabilities**.
- Produces normalized fields: description, industry taxonomy, company size, target markets, and brand voice.

### 2. ICP Generator (`sellbird/icp_generator.py`)
- Analyzes the specific product's features, benefits, and objection playbooks.
- Synthesizes targeted buyer personas, acute operational pain points, high-intent buying signals, and qualification gates.
- Associates the resulting ICP directly with `product_id`.

---

## 5. REST API Reference

All endpoints are hosted on the FastAPI application (`sellbird/products_router.py`) mounted in `sellbird/dashboard.py`.

### Businesses

#### `POST /api/businesses`
Create a new business profile.
```json
{
  "name": "Acme Cyber",
  "description": "Cloud security automation platform",
  "website": "https://acmecyber.example.com",
  "industry": "Cybersecurity",
  "company_size": "50-200",
  "business_location": "Austin, TX",
  "target_markets": ["North America", "UK"],
  "brand_voice": "Authoritative and concise"
}
```

#### `GET /api/businesses`
List all business profiles.

#### `GET /api/businesses/{business_id}`
Get details for a specific business.

#### `PUT /api/businesses/{business_id}`
Update fields on a business profile.

#### `DELETE /api/businesses/{business_id}?force={true|false}`
Delete a business profile. Protected by foreign-key safeguard: returns HTTP 400 if products are linked, unless `force=true`.

---

### Products

#### `POST /api/products`
Create a product linked to a business.
```json
{
  "business_id": "b1a2c3d4e5f6",
  "name": "CloudGuard Pro",
  "description": "Continuous container security compliance",
  "category": "DevSecOps",
  "pricing": "$500/cluster/month",
  "key_features": ["eBPF telemetry", "Automated remediation"],
  "key_benefits": ["Zero overhead", "Audit-ready reports in 1 click"],
  "unique_selling_points": ["No kernel patching required"],
  "target_industries": ["Fintech", "Healthtech"],
  "objection_responses": {
    "Too expensive": "Saves 20 engineer hours/week in compliance prep."
  },
  "offer": {
    "type": "Free POC",
    "cta": "Book a 15-minute architecture walk"
  }
}
```

#### `GET /api/products?business_id={id}`
List products, optionally filtered by `business_id`.

#### `GET /api/products/{product_id}`
Get product details.

#### `PUT /api/products/{product_id}`
Update product fields.

#### `DELETE /api/products/{product_id}?force={true|false}`
Delete a product. Protected by safeguard: returns HTTP 400 if prospects are linked, unless `force=true`.

#### `GET /api/products/{product_id}/prospects?status={status}`
Get prospects assigned to this specific product.

#### `GET /api/products/{product_id}/campaigns`
Get outreach campaigns targeting this specific product.

---

### Ideal Customer Profiles (ICPs)

#### `POST /api/icps`
Create an ICP linked to a product.
```json
{
  "product_id": "p9a8b7c6d5e4",
  "name": "Fintech DevSecOps Leads",
  "description": "Engineering and security leaders at high-growth fintechs",
  "target_industries": ["Fintech", "Payment Processing"],
  "company_sizes": ["100-1000"],
  "target_geographies": ["US", "Canada", "UK"],
  "job_titles": ["VP Engineering", "Head of Security", "DevSecOps Lead"],
  "buyer_personas": ["Security leader balancing fast deployments with compliance audits"],
  "pain_points": ["Manual SOC2 evidence gathering", "Developer friction"],
  "buying_signals": ["Hiring compliance engineers", "Raised Series B"],
  "qualification_criteria": ["Using Kubernetes in AWS or GCP"]
}
```

#### `GET /api/icps?product_id={id}`
List ICPs, optionally filtered by `product_id`.

#### `GET /api/icps/{icp_id}`
Get ICP details.

#### `PUT /api/icps/{icp_id}`
Update ICP fields.

#### `DELETE /api/icps/{icp_id}`
Delete an ICP.

---

### Local AI Synthesis

#### `POST /api/businesses/generate`
Synthesizes a structured business profile from raw inputs using local Ollama inference.
```json
{
  "name": "DataScale",
  "website": "https://datascale.example.com",
  "description": "High performance vector cache for LLM applications",
  "industry": "Data Infrastructure",
  "company_facts": ["Built on Rust", "Open-source core with enterprise cloud"]
}
```

#### `POST /api/icps/generate`
Synthesizes a product-tailored ICP from product specifications using local Ollama inference.
```json
{
  "product_id": "p9a8b7c6d5e4",
  "target_market_hint": "Series A-B AI startups with latency bottlenecks"
}
```

---

## 6. Verification and Test Suite

The multi-product architecture is backed by an automated test suite:

- **`tests/test_multi_product.py`**:
  - Business profile CRUD operations
  - Multiple products per business
  - Multiple ICPs per product
  - Product retrieval and updates
  - Relational deletion safeguards
  - Product-specific prospect and campaign filtering
  - Migration v18 idempotency and legacy data backfill
  - Local AI Business Profile generation
  - Local AI ICP generation
  - Zero cross-product context leakage during agent execution
- **`tests/test_product_api.py`**:
  - End-to-end FastAPI endpoint tests using `TestClient`
  - Authentication checks and token validation
  - Request schema validation and HTTP 400 error handling
  - Foreign key integrity guards and force delete flags
  - AI generation endpoint contract validation

Run the test suite with:
```bash
pytest tests/test_multi_product.py tests/test_product_api.py -v
```
