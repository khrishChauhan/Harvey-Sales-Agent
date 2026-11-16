# Harvey Local AI Sales Agent — Application User Guide

A complete operational guide for managing multi-product sales, prospect intelligence, AI-assisted outreach, and local inference telemetry through the Harvey dashboard.

---

## 1. Starting the Application

Harvey runs as a local web application powered by FastAPI and a Vanilla JavaScript/HTML/CSS dashboard. All AI inference and semantic embeddings execute strictly on your local hardware via **Ollama** and **Sentence Transformers**.

### Prerequisites
1. **Ollama running locally**: Ensure Ollama is running on loopback (`http://127.0.0.1:11434`) with the `qwen3:4b` model pulled:
   ```bash
   ollama pull qwen3:4b
   ```
2. **Local Embedding Model**: Verify the lightweight sentence-transformer model (`all-MiniLM-L6-v2`) is cached locally.

### Launching the Dashboard
Launch the dashboard via the Sellbird CLI:
```bash
sellbird dashboard
```
Or start the ASGI server directly:
```bash
uvicorn sellbird.main:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
```
http://127.0.0.1:8000/dashboard
```

---

## 2. Business Profile Management

Manage one or multiple company entities that offer products or services.

### Accessing Business Profiles
1. In the sidebar navigation, click **Products & Businesses**.
2. Click the **Businesses** subtab.

### Creating or Editing a Business Profile
1. Click **+ New Business Profile**.
2. Fill in the business facts:
   - **Business Name**: Legal or trading name of your business.
   - **Domain / Website**: Primary company URL (e.g., `acmecloud.io`).
   - **Description**: Summary of your business operations.
   - **Target Market**: Primary industry focus and segments.
   - **Brand Voice**: Desired tone of voice (e.g., `Concise, technical, empathetic`).
3. Click **Save Business Profile**.

### AI-Assisted Business Generation
To draft a business profile quickly from minimal information:
1. In the Business Profile modal, enter at least a **Company Name** and a brief **Industry/Description** hint.
2. Click **Generate with Local AI**.
3. Local Ollama (`qwen3:4b`) will analyze the hint and suggest descriptions, target markets, and brand voices.
4. **Human Review**: AI suggestions are populated into the form fields. Review, edit, or customize any text before clicking **Save Business Profile**. User-supplied facts are strictly preserved.

---

## 3. Product & Offering Management

Harvey supports multiple products under each business profile. Each product maintains its own independent positioning, pricing, objection handlers, and active state.

### Managing Products
1. Navigate to **Products & Businesses** > **Products** subtab.
2. The list displays all registered products, grouped by business, along with pricing, USPs, and active status.

### Creating a Product
1. Click **+ New Product**.
2. Enter product details:
   - **Parent Business**: Select the business profile this product belongs to.
   - **Product Name**: Specific name of the product or service.
   - **Pricing**: Pricing model (e.g., `$199/month`, `$5,000/year`).
   - **Primary Offer Entry**: The initial hook or offer (e.g., `Free 14-day telemetry audit`).
   - **Description**: Detailed product summary.
   - **Key Features**: Comma-separated list of architectural capabilities.
   - **Value Benefits**: Comma-separated list of measurable outcomes.
   - **Unique Selling Points (USPs)**: Key differentiators over alternatives.
   - **Objection Handlers (JSON)**: Key-value map of objection responses (e.g., `{"pricing": "Pays for itself in 30 days"}`).
3. Click **Save Product**.

### Selecting the Active Product
1. Use the **Active Product Selector** bar at the top of the screen or within the Products table.
2. Selecting a product immediately establishes the global context for:
   - Prospect filtering and scoring.
   - ICP alignment.
   - Cold email hook synthesis.
   - Campaign analytics.

---

## 4. Ideal Customer Profile (ICP) Management

ICPs define the exact target criteria for a specific product, ensuring AI prospecting targets high-conversion decision-makers.

### Creating an ICP
1. Navigate to **Products & Businesses** > **Target ICPs** subtab.
2. Click **+ New ICP**.
3. Associate the ICP with an existing product and define:
   - **ICP Name**: Persona label (e.g., `FinTech CISOs & SecOps`).
   - **Target Industries**: Comma-separated (e.g., `FinTech, Cybersecurity, Banking`).
   - **Company Sizes**: Comma-separated (e.g., `50-250, 250-1000`).
   - **Geographies**: Target regions (e.g., `North America, Western Europe`).
   - **Target Titles / Personas**: Comma-separated job titles (e.g., `CISO, VP Security, SecOps Director`).
   - **Core Pain Points**: Key problems your product resolves.
   - **High-Value Buying Signals**: Triggers that indicate purchase readiness (e.g., `Hiring compliance analysts, Series B funding, SOC2 audits`).

### AI-Assisted ICP Generation
1. In the ICP modal, select the **Product**.
2. Click **Generate ICP with Local AI**.
3. Local Ollama synthesizes the product's features, benefits, and market positioning to output recommended personas, pain points, and buying signals.
4. Review and adjust all generated criteria before saving.

---

## 5. Prospect Intelligence & Research Verification

The **Prospect Intelligence Drawer** provides full transparency into how Harvey evaluates and scores leads.

### Opening Prospect Intelligence
1. Navigate to **Prospects** in the main navigation.
2. Filter prospects by product using the product dropdown.
3. Click the **Intelligence** button next to any prospect row.

### Understanding the Intelligence View

#### 1. Header & Semantic Match
- Displays contact name, title, company, email verification status, and assigned product/ICP.
- **Semantic ICP Match**: Shows cosine similarity (0–100%) between the prospect profile and the product's ICP criteria computed locally via `all-MiniLM-L6-v2`.

#### 2. Explainable Prospect Score (0–100)
Harvey never outputs arbitrary scores. Every score is fully explainable and strictly bounded between 0 and 100:
- **Industry Fit (Max 20 pts)**: Exact or related industry match.
- **Job Title Fit (Max 30 pts)**: Seniority and functional relevance.
- **Email Deliverability (Max 25 pts)**: Verified MX records, SMTP handshake, or trusted inbox status.
- **Observed Buying Signals (Max 25 pts)**: Active hiring, funding, tech stack triggers, or regulatory deadlines.
- **Semantic ICP Boost (Max 15 pts)**: High semantic similarity boost (applied only to prospects with valid title/industry foundations, capped at 100).
- Click **Re-calculate Score** to rerun the local scoring engine with any updated observations.

#### 3. Verified Facts vs. AI Interpretations
Harvey enforces strict separation between verified ground truths and AI inference:
- **Verified Facts (`badge-verified`)**: Scraped and confirmed data points (funding announcements, press releases, job board postings) accompanied by verified **Source URLs**.
- **Buying Signals & AI Insights (`badge-ai`)**: Local model interpretations with confidence scores (e.g., `High (0.85)`).
- **Evidence-Grounded Problem Hook**: A concise business problem hook synthesized by local Ollama strictly from the observed facts. Never treats speculative inferences as facts.

---

## 6. Outreach Generation & Human Approval Workflow

Harvey enforces a strict **Human-in-the-Loop** safety architecture. The AI drafts messages, but emails are never sent without explicit human authorization.

### Generating a Draft
1. From the Prospect Intelligence modal, click **Draft Outreach Email** (or use the Action button in the Prospects table).
2. The modal loads:
   - **Subject Line**: Generated from the prospect's verified problem statement and active product offer.
   - **Body Text**: Personalized 3-paragraph outreach referencing specific verified facts, the prospect's role, and an unobtrusive call to action.
3. **Edit Draft**: Full rich text/textarea editing allows you to adjust tone, add custom personal notes, or modify the subject line.

### Staging to Outbox
1. Click **Stage to Outbox (Require Review)**.
2. The message is committed to SQLite with status `pending_review`.
3. It appears in the **Outbox** tab with an amber `Pending Approval` badge.
4. It will remain staged indefinitely until a user explicitly reviews and clicks **Approve & Send**. Autopilot auto-sending of unapproved drafts is strictly disabled.

### Message & Reply Classifications
When replies are received, Harvey classifies them into standard buckets:
- `interested`: Requests a demo, pricing, or call.
- `not_interested`: Explicit polite rejection.
- `out_of_office`: Auto-responder / vacation notice.
- `unsubscribe`: Opt-out request (prospect status permanently moved to `opted_out`).
- `wrong_person`: Referral to another colleague.

---

## 7. Sales Analytics

Gain real-time visibility into multi-product pipeline performance, qualification rates, and signal trends.

### Accessing Analytics
1. In the sidebar navigation, click **Analytics**.
2. Filter by **All Products** or select an individual product from the dropdown.

### Metrics & Visualizations
- **Total Prospects**: Total contacts in the active scope.
- **Qualified Prospects**: Prospects meeting or exceeding the qualification threshold (`score >= 60`).
- **Qualification Rate**: Percentage of leads matching qualification criteria.
- **Active Campaigns**: Number of active outreach sequences.
- **Pipeline Stage Funnel**: Breakdown of prospects across `new`, `researched`, `scored`, `queued`, `contacted`, `replied`, and `converted`.
- **Product Breakdown Table**: Comparative metrics showing prospect volume, qualified count, active campaigns, and conversion counts across all business products.
- **Buying Signal Distribution**: Frequency and categorization of detected signals across prospects (e.g., `HIRING_EXPANSION`, `FUNDING_ROUND`, `REGULATORY_COMPLIANCE`).

---

## 8. Local AI & Hardware Telemetry

Monitor your local inference engine and hardware consumption directly from the **Usage** dashboard.

### Accessing AI Status
1. Navigate to **Usage** in the sidebar navigation.
2. Locate the **Local AI & Hardware Infrastructure** card.

### Displayed Telemetry
1. **Inference Provider**: Verified as `Ollama (Local Loopback http://127.0.0.1:11434)`.
2. **Primary LLM Model**: `qwen3:4b` (Apache 2.0 open-weights).
3. **Embedding Engine**: `sentence-transformers (all-MiniLM-L6-v2)`.
4. **Physical RAM**: Real Win32 physical memory capacity and currently available RAM (e.g., `16.0 GB Total / 7.2 GB Available`).
5. **NVIDIA GPU & VRAM**: Detected GPU hardware and total VRAM (e.g., `NVIDIA GeForce RTX 2050 (4.0 GB VRAM)`).
6. **Local Inference Logs**:
   - Total local prompt completions executed.
   - Total tokens generated offline.
   - Average inference latency (ms).
   - Real-time audit log of recent agent calls with model, latency, and token metrics.

---

## 9. Architectural Integrity & Safety Guarantees

| Guarantee | Enforcement Mechanism |
| :--- | :--- |
| **Zero Cloud AI Leakage** | All API keys for external LLM providers (OpenAI, Anthropic, Gemini) are blocked at configuration validation time. Any non-loopback host is rejected by the runtime. |
| **Mandatory Human Approval** | New emails stage as `pending_review` in SQLite. No automated background process may transition an unapproved message to `sent`. |
| **Strict 0–100 Scoring** | Normalization algorithms in `sellbird/scoring.py` ensure additive factors and semantic boosts never exceed 100 points. |
| **Ground Truth Attribution** | Every research observation must have a verified `source_url` before being classified as a verified fact. AI hypotheses are explicitly tagged with `badge-ai`. |
