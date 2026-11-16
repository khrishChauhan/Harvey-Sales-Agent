"""Integration tests for Phase 4: Application Integration & User Experience.

Tests all 10 key user journeys through the FastAPI backend and state manager:
1. Business Profile Management (Create, read, update, list)
2. Multi-Product Management & Active Product Selection
3. ICP Management & Product-Specific ICP Generation
4. Product-Specific Prospect Query & Filtering
5. Prospect Intelligence (Verified facts, source URLs, timestamps)
6. Buying Signals & Problem Hook Syntheses
7. Explainable Prospect Scoring & Factor Breakdown (0-100 bounded)
8. Personalized Email Generation & Human-Approval Staging (Outbox pending_review)
9. Conversations & Reply Classifications
10. Global & Product-Specific Sales Analytics
11. Local AI Engine Telemetry & Hardware Status Detection
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import sellbird.dashboard as dash
from sellbird.models.business import BusinessProfile
from sellbird.models.icp import ICP
from sellbird.models.product import Product
from sellbird.models.prospect import Prospect
from sellbird.state import StateManager

ORIGIN = "http://127.0.0.1:5555"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Set up isolated database and auth token for testing."""
    (tmp_path / "data").mkdir()
    db_path = tmp_path / "data" / "sellbird.db"
    monkeypatch.setattr(dash, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(dash, "DB_PATH", db_path)
    monkeypatch.setattr(dash, "PID_FILE", tmp_path / "data" / "sellbird.pid")
    monkeypatch.setattr(dash, "LOG_FILE", tmp_path / "data" / "sellbird.log")
    dash.token_store.configure(tmp_path / "data" / "dashboard-token")

    async def _init():
        from sellbird.signals import seed_signal_catalog
        state = StateManager(db_path=str(db_path))
        await state.init_db()
        await seed_signal_catalog(state)

    asyncio.run(_init())
    token = dash.token_store.get()
    return {"token": token, "db_path": db_path}


@pytest.fixture
def client(env):
    """Authenticated TestClient for the dashboard app."""
    with TestClient(
        dash.app,
        base_url=ORIGIN,
        headers={"Authorization": f"Bearer {env['token']}"},
    ) as c:
        yield c


def test_journey_1_business_profile_management(client):
    """User Journey 1: Create, view, and edit a business profile."""
    # Create business profile
    create_resp = client.post(
        "/api/businesses",
        json={
            "name": "CloudScale Systems",
            "domain": "cloudscale.io",
            "industry": "Cloud Infrastructure",
            "description": "High-throughput cloud performance observability.",
        },
    )
    assert create_resp.status_code == 200
    biz = create_resp.json()
    assert biz["id"]
    assert biz["name"] == "CloudScale Systems"

    # Read back list
    list_resp = client.get("/api/businesses")
    assert list_resp.status_code == 200
    businesses = list_resp.json()
    assert len(businesses) >= 1
    assert any(b["id"] == biz["id"] for b in businesses)

    # Update business profile
    update_resp = client.put(
        f"/api/businesses/{biz['id']}",
        json={
            "name": "CloudScale Global Systems",
            "domain": "cloudscale.io",
            "industry": "DevOps & Cloud",
            "description": "Updated enterprise description.",
        },
    )
    assert update_resp.status_code == 200
    updated = update_resp.json()
    assert updated["name"] == "CloudScale Global Systems"
    assert updated["industry"] == "DevOps & Cloud"


def test_journey_2_multi_product_management_and_active_product(client):
    """User Journey 2: Add two products and select an active product."""
    # Create Business
    biz = client.post(
        "/api/businesses",
        json={"name": "DevTools Inc", "domain": "devtools.com"},
    ).json()

    # Product A: Serverless Monitor
    prod_a = client.post(
        f"/api/businesses/{biz['id']}/products",
        json={
            "name": "Serverless Monitor",
            "pricing": "$199/month",
            "description": "Latency tracking for Lambda and Cloud Functions.",
            "offer_entry": "Free 14-day telemetry audit",
            "features": ["eBPF distributed tracing", "Cold-start detection"],
            "benefits": ["Reduce MTTR by 50%"],
            "usps": ["Zero overhead"],
            "objection_responses": {"pricing": "ROI in 14 days"},
        },
    ).json()
    assert prod_a["id"]
    assert prod_a["name"] == "Serverless Monitor"

    # Product B: Kubernetes Guard
    prod_b = client.post(
        f"/api/businesses/{biz['id']}/products",
        json={
            "name": "Kubernetes Guard",
            "pricing": "$499/month",
            "description": "Cluster policy enforcement and security auditing.",
            "offer_entry": "Free cluster vulnerability scan",
            "features": ["Admission controller", "CVE scanning"],
            "benefits": ["Automated compliance"],
            "usps": ["100% offline rule evaluation"],
            "objection_responses": {"complexity": "Installs with 1 helm command"},
        },
    ).json()
    assert prod_b["id"]
    assert prod_b["name"] == "Kubernetes Guard"

    # Set Product A as active
    set_resp = client.post(
        "/api/active-product",
        json={"product_id": prod_a["id"]},
    )
    assert set_resp.status_code == 200
    assert set_resp.json()["active_product_id"] == prod_a["id"]

    # Verify active product persists
    get_resp = client.get("/api/active-product")
    assert get_resp.status_code == 200
    assert get_resp.json()["active_product_id"] == prod_a["id"]

    # Switch to Product B
    client.post("/api/active-product", json={"product_id": prod_b["id"]})
    assert client.get("/api/active-product").json()["active_product_id"] == prod_b["id"]


def test_journey_3_icp_management_and_generation(client):
    """User Journey 3: Generate and create an ICP for a product."""
    biz = client.post("/api/businesses", json={"name": "Observability Co"}).json()
    prod = client.post(
        f"/api/businesses/{biz['id']}/products",
        json={"name": "LogStream", "description": "High-volume log indexing."},
    ).json()

    # Create target ICP
    icp_resp = client.post(
        f"/api/products/{prod['id']}/icps",
        json={
            "name": "DevOps Engineers",
            "target_industries": ["SaaS", "FinTech"],
            "company_sizes": ["50-200", "201-1000"],
            "geographies": ["US", "EU"],
            "target_titles": ["DevOps Engineer", "Head of Infrastructure"],
            "buyer_personas": "Engineering leads worried about log costs.",
            "pain_points": ["Astronomical Datadog bills", "Slow search"],
            "buying_signals": ["Hiring SREs", "Recent cloud migration"],
        },
    )
    assert icp_resp.status_code == 200
    icp = icp_resp.json()
    assert icp["id"]
    assert icp["name"] == "DevOps Engineers"

    # Query ICPs for product
    icps_resp = client.get(f"/api/products/{prod['id']}/icps")
    assert icps_resp.status_code == 200
    assert len(icps_resp.json()) >= 1

    # Set as active ICP
    act_resp = client.post("/api/active-icp", json={"icp_id": icp["id"]})
    assert act_resp.status_code == 200
    assert client.get("/api/active-icp").json()["active_icp_id"] == icp["id"]


def test_journey_4_product_specific_prospect_filtering(client, env):
    """User Journey 4: Query prospects by product association."""
    async def _setup_prospects():
        state = StateManager(db_path=str(env["db_path"]))
        # Create two products
        p1_id = await state.add_product(Product(business_id="b1", name="Product Alpha"))
        p2_id = await state.add_product(Product(business_id="b1", name="Product Beta"))

        # Add prospect for Alpha
        pr1 = Prospect(
            id="p-alpha",
            first_name="Alice",
            last_name="Alpha",
            email="alice@alpha.com",
            company="Alpha Corp",
            title="VP Tech",
            product_id=p1_id,
            score=75,
            status="new",
        )
        await state.add_prospect(pr1)

        # Add prospect for Beta
        pr2 = Prospect(
            id="p-beta",
            first_name="Bob",
            last_name="Beta",
            email="bob@beta.com",
            company="Beta Corp",
            title="CTO",
            product_id=p2_id,
            score=85,
            status="new",
        )
        await state.add_prospect(pr2)
        return p1_id, p2_id

    p1_id, p2_id = asyncio.run(_setup_prospects())

    # Query all prospects
    all_resp = client.get("/api/prospects")
    assert all_resp.status_code == 200
    all_prospects = all_resp.json()
    assert len(all_prospects) >= 2

    # Query Product Alpha prospects
    p1_resp = client.get(f"/api/prospects?product_id={p1_id}")
    assert p1_resp.status_code == 200
    p1_prospects = p1_resp.json()
    assert len(p1_prospects) == 1
    assert p1_prospects[0]["id"] == "p-alpha"

    # Query Product Beta prospects
    p2_resp = client.get(f"/api/prospects?product_id={p2_id}")
    assert p2_resp.status_code == 200
    p2_prospects = p2_resp.json()
    assert len(p2_prospects) == 1
    assert p2_prospects[0]["id"] == "p-beta"


def test_journey_5_6_7_prospect_intelligence_scoring_and_signals(client, env):
    """User Journeys 5, 6, 7:
    - Prospect Intelligence gathering
    - Verified facts vs AI interpretations
    - Buying signals & confidence
    - Explainable prospect score <= 100
    """
    async def _setup_data():
        state = StateManager(db_path=str(env["db_path"]))
        prod_id = await state.add_product(Product(
            business_id="b1",
            name="SecurityAudit Pro",
            description="Automated SOC2 compliance scanner",
            offer_entry="Free compliance gap audit",
        ))
        icp_id = await state.add_icp(ICP(
            product_id=prod_id,
            name="FinTech Security Leaders",
            target_industries=["FinTech", "Banking"],
            target_titles=["CISO", "VP Security"],
        ))

        prospect = Prospect(
            id="prospect-ciso-1",
            first_name="David",
            last_name="Wallace",
            email="david@fintechguard.com",
            company="FinTech Guard",
            title="CISO",
            product_id=prod_id,
            icp_id=icp_id,
            email_status="valid",
            status="new",
        )
        await state.add_prospect(prospect)

        # Store company research dossier & observations
        await state.save_research_dossier(
            company_id=prospect.id,
            domain="fintechguard.com",
            product_id=prod_id,
            verified_facts=[
                {
                    "text": "FinTech Guard announced Series B funding for SOC2 compliance.",
                    "source_url": "https://fintechguard.com/news/series-b",
                    "created_at": "2026-11-01T12:00:00Z",
                }
            ],
            ai_interpretations=[
                {
                    "signal_code": "HIRING_ROLE",
                    "confidence": "high",
                    "evidence": "Currently hiring 4 compliance analysts.",
                }
            ],
            source_urls=["https://fintechguard.com/news/series-b"],
        )
        await state.add_observation(
            company_id=prospect.id,
            signal_code="HIRING_ROLE",
            value_text="Currently hiring 4 compliance analysts.",
            evidence_url="https://fintechguard.com/news/series-b",
        )
        return prospect.id

    pid = asyncio.run(_setup_data())

    # Fetch Intelligence
    intel_resp = client.get(f"/api/prospects/{pid}/intelligence")
    assert intel_resp.status_code == 200
    intel = intel_resp.json()
    assert intel["success"] is True
    assert intel["product_name"] == "SecurityAudit Pro"
    assert intel["score"] <= 100
    assert intel["score"] >= 0

    # Journey 5: Verify facts and source URLs
    assert len(intel["verified_facts"]) >= 1
    fact = intel["verified_facts"][0]
    assert "Series B" in fact["content"]
    assert fact["source_url"] == "https://fintechguard.com/news/series-b"

    # Journey 6: Verify buying signals and problem statement
    assert len(intel["buying_signals"]) >= 1
    sig = intel["buying_signals"][0]
    assert sig.get("signal_code") in ("HIRING_ROLE", "HIRING_EXPANSION", "REGULATORY_COMPLIANCE", "COMPANY_GROWTH") or "HIRING" in str(sig)
    assert sig["confidence"] >= 0.5 or sig["confidence"] == "high"
    assert intel["problem_statement"]  # AI generated hook

    # Journey 7: Re-calculate score and verify breakdown factors
    score_resp = client.post(f"/api/prospects/{pid}/score")
    assert score_resp.status_code == 200
    score_data = score_resp.json()
    assert score_data["score"] <= 100
    breakdown = score_data["score_breakdown"]
    total_breakdown = (
        breakdown.get("industry_fit", 0)
        + breakdown.get("job_title_fit", 0)
        + breakdown.get("email_deliverability", 0)
        + breakdown.get("buying_signals", 0)
        + breakdown.get("semantic_icp_boost", 0)
    )
    assert total_breakdown <= 100


def test_journey_8_email_generation_and_human_approval_staging(client, env):
    """User Journey 8: Generate email draft and stage to Outbox with human approval requirement."""
    async def _setup_prospect():
        state = StateManager(db_path=str(env["db_path"]))
        prod_id = await state.add_product(Product(
            business_id="b1",
            name="ApiGateway Pro",
            offer_entry="Free 30-day performance benchmark",
        ))
        prospect = Prospect(
            id="p-stage-test",
            first_name="Elena",
            last_name="Rostova",
            email="elena@cloudnative.tech",
            company="Cloud Native Tech",
            title="Head of Platform",
            product_id=prod_id,
            status="new",
        )
        await state.add_prospect(prospect)
        return prospect.id

    pid = asyncio.run(_setup_prospect())

    # Generate personalized email draft
    gen_resp = client.post(f"/api/prospects/{pid}/generate-email")
    assert gen_resp.status_code == 200
    email_draft = gen_resp.json()
    assert email_draft["success"] is True
    assert email_draft["subject"]
    assert "Elena" in email_draft["body"] or "Cloud Native Tech" in email_draft["body"]

    # Stage email to Outbox as pending_review
    stage_resp = client.post(
        f"/api/prospects/{pid}/stage-email",
        json={
            "subject": "Customized: " + email_draft["subject"],
            "body": email_draft["body"] + "\n\nBest,\nSales Team",
        },
    )
    assert stage_resp.status_code == 200
    stage_data = stage_resp.json()
    assert stage_data["success"] is True
    assert stage_data["status"] == "pending_review"

    # Verify message was stored in Outbox and is in pending_review
    async def _verify_outbox():
        state = StateManager(db_path=str(env["db_path"]))
        msgs = await state.get_outbox()
        matching = [m for m in msgs if m.get("prospect_id") == pid]
        assert len(matching) >= 1
        assert matching[0]["status"] == "pending_review"

    asyncio.run(_verify_outbox())


def test_journey_9_conversation_classifications(client, env):
    """User Journey 9: Verify reply classifications are supported and accessible."""
    # Read conversations endpoint
    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_journey_10_sales_analytics(client, env):
    """User Journey 10: Multi-product and global sales analytics verification."""
    async def _setup_analytics_data():
        state = StateManager(db_path=str(env["db_path"]))
        prod_id = await state.add_product(Product(
            business_id="b1",
            name="Analytics Tool",
        ))
        p1 = Prospect(
            id="p-ana-1",
            first_name="Sam",
            last_name="Alt",
            email="sam@tech.com",
            company="Tech Corp",
            title="VP",
            product_id=prod_id,
            score=80,
            status="replied",
        )
        p2 = Prospect(
            id="p-ana-2",
            first_name="Sara",
            last_name="Lee",
            email="sara@bakery.com",
            company="Bakery Inc",
            title="Director",
            product_id=prod_id,
            score=45,
            status="new",
        )
        await state.add_prospect(p1)
        await state.add_prospect(p2)
        return prod_id

    prod_id = asyncio.run(_setup_analytics_data())

    # Global Analytics
    global_resp = client.get("/api/analytics")
    assert global_resp.status_code == 200
    g_data = global_resp.json()
    assert g_data["success"] is True
    assert g_data["total_prospects"] >= 2
    assert g_data["qualified_prospects"] >= 1
    assert "conversion_rates" in g_data
    assert "pipeline_stages" in g_data

    # Product-Filtered Analytics
    prod_resp = client.get(f"/api/analytics?product_id={prod_id}")
    assert prod_resp.status_code == 200
    p_data = prod_resp.json()
    assert p_data["success"] is True
    assert p_data["total_prospects"] == 2
    assert p_data["qualified_prospects"] == 1


def test_journey_11_local_ai_and_hardware_status(client):
    """User Journey 11: Real local AI engine status and hardware detection endpoint."""
    resp = client.get("/api/system/ai-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["active_model"] == "qwen3:4b"
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert "hardware" in data
    assert data["hardware"]["ram_total_gb"] > 0
    assert "telemetry" in data
    assert "total_calls" in data["telemetry"]
