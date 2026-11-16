"""API Integration Tests for Multi-Product Architecture Endpoints.

Tests:
- Business profile CRUD (/api/businesses)
- Product CRUD (/api/products) and business linking
- ICP CRUD (/api/icps) and product linking
- Product-specific prospects and campaigns endpoints
- AI generation endpoints (/api/businesses/generate, /api/icps/generate)
- Validation, foreign-key checks, and deletion safeguards
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
        state = StateManager(db_path=str(db_path))
        await state.init_db()

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


def test_business_crud_api(client):
    """Test full CRUD lifecycle for business profiles via REST API."""
    # 1. Create Business
    payload = {
        "name": "Acme Cloud Solutions",
        "description": "Enterprise cloud observability platform",
        "website": "https://acmecloud.example.com",
        "industry": "Cloud Infrastructure",
        "company_size": "50-200",
        "business_location": "Berlin, Germany",
        "target_markets": ["EMEA", "North America"],
        "brand_voice": "Technical, authoritative",
    }
    resp = client.post("/api/businesses", json=payload)
    assert resp.status_code == 200, resp.text
    biz = resp.json()
    biz_id = biz["id"]
    assert biz["name"] == "Acme Cloud Solutions"
    assert biz["industry"] == "Cloud Infrastructure"

    # 2. List Businesses
    resp = client.get("/api/businesses")
    assert resp.status_code == 200
    all_biz = resp.json()
    assert any(b["id"] == biz_id for b in all_biz)

    # 3. Get Specific Business
    resp = client.get(f"/api/businesses/{biz_id}")
    assert resp.status_code == 200
    assert resp.json()["website"] == "https://acmecloud.example.com"

    # 4. Update Business
    update_payload = {"company_size": "100-500", "brand_voice": "Consultative and crisp"}
    resp = client.put(f"/api/businesses/{biz_id}", json=update_payload)
    assert resp.status_code == 200
    assert resp.json()["company_size"] == "100-500"
    assert resp.json()["brand_voice"] == "Consultative and crisp"

    # 5. Delete Business
    resp = client.delete(f"/api/businesses/{biz_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # 6. Verify 404 after deletion
    resp = client.get(f"/api/businesses/{biz_id}")
    assert resp.status_code == 404


def test_product_crud_and_safeguards_api(client):
    """Test product creation, retrieval, filtering, and deletion safeguards via API."""
    # Create business first
    b_resp = client.post(
        "/api/businesses",
        json={"name": "Logix Corp", "industry": "DevSecOps"},
    )
    biz_id = b_resp.json()["id"]

    # Attempt to create product with nonexistent business_id -> 400
    bad_resp = client.post(
        "/api/products",
        json={"business_id": "nonexistent-biz-id", "name": "Fake Tool"},
    )
    assert bad_resp.status_code == 400

    # Create Product A
    p1_resp = client.post(
        "/api/products",
        json={
            "business_id": biz_id,
            "name": "Logix Guard",
            "category": "Kubernetes Security",
            "pricing": "$500/node/month",
            "key_features": ["Runtime enforcement", "Zero-trust network policy"],
            "key_benefits": ["Prevent container escapes", "Continuous compliance"],
            "unique_selling_points": ["eBPF without kernel patching"],
            "target_industries": ["Fintech", "Healthtech"],
            "objection_responses": {"Too complex": "Agentless eBPF deploys in 5 minutes."},
            "offer": {"type": "14-day free pilot"},
        },
    )
    assert p1_resp.status_code == 200
    p1 = p1_resp.json()
    p1_id = p1["id"]

    # Create Product B
    p2_resp = client.post(
        "/api/products",
        json={
            "business_id": biz_id,
            "name": "Logix Cost Optimizer",
            "category": "FinOps",
        },
    )
    assert p2_resp.status_code == 200
    p2_id = p2_resp.json()["id"]

    # List products filtered by business_id
    resp = client.get(f"/api/products?business_id={biz_id}")
    assert resp.status_code == 200
    products = resp.json()
    assert len(products) == 2
    ids = [p["id"] for p in products]
    assert p1_id in ids and p2_id in ids

    # Update Product A
    resp = client.put(f"/api/products/{p1_id}", json={"pricing": "$650/node/month"})
    assert resp.status_code == 200
    assert resp.json()["pricing"] == "$650/node/month"

    # Business deletion should be blocked while products exist unless force=True
    del_biz_blocked = client.delete(f"/api/businesses/{biz_id}")
    assert del_biz_blocked.status_code == 400
    assert "Cannot delete business" in del_biz_blocked.json()["detail"]

    # Delete product without force
    del_prod = client.delete(f"/api/products/{p2_id}")
    assert del_prod.status_code == 200

    # Verify product is deleted
    assert client.get(f"/api/products/{p2_id}").status_code == 404


def test_icp_crud_api(client):
    """Test ICP creation, retrieval, filtering, and deletion via API."""
    # Setup business & product
    b_resp = client.post("/api/businesses", json={"name": "DataMesh Co"})
    biz_id = b_resp.json()["id"]
    p_resp = client.post("/api/products", json={"business_id": biz_id, "name": "DataPipeline"})
    prod_id = p_resp.json()["id"]

    # Attempt to create ICP for nonexistent product -> 400
    bad_icp = client.post("/api/icps", json={"product_id": "ghost-prod", "name": "Fake ICP"})
    assert bad_icp.status_code == 400

    # Create valid ICP
    icp_payload = {
        "product_id": prod_id,
        "name": "Enterprise Data Leaders",
        "description": "VP of Data & Head of Analytics at Series B+ tech firms",
        "target_industries": ["B2B SaaS", "E-commerce"],
        "company_sizes": ["100-1000"],
        "target_geographies": ["US", "Canada"],
        "job_titles": ["VP Data", "Head of Data Engineering", "Chief Data Officer"],
        "buyer_personas": ["Technical executive focused on data pipeline uptime"],
        "pain_points": ["Airflow cluster maintenance", "Silent schema breaks"],
        "buying_signals": ["Hiring dbt analytics engineer", "Raised Series B"],
        "qualification_criteria": ["Snowflake or BigQuery stack"],
    }
    resp = client.post("/api/icps", json=icp_payload)
    assert resp.status_code == 200
    icp = resp.json()
    icp_id = icp["id"]
    assert icp["name"] == "Enterprise Data Leaders"
    assert icp["product_id"] == prod_id

    # List ICPs filtered by product_id
    resp = client.get(f"/api/icps?product_id={prod_id}")
    assert resp.status_code == 200
    icps = resp.json()
    assert len(icps) == 1
    assert icps[0]["id"] == icp_id

    # Get specific ICP
    resp = client.get(f"/api/icps/{icp_id}")
    assert resp.status_code == 200
    assert "Snowflake or BigQuery stack" in resp.json()["qualification_criteria"]

    # Update ICP
    resp = client.put(f"/api/icps/{icp_id}", json={"company_sizes": ["200-2000"]})
    assert resp.status_code == 200
    assert resp.json()["company_sizes"] == ["200-2000"]

    # Delete ICP
    resp = client.delete(f"/api/icps/{icp_id}")
    assert resp.status_code == 200
    assert client.get(f"/api/icps/{icp_id}").status_code == 404


def test_product_prospects_and_campaigns_api(client, env):
    """Test retrieval of product-specific prospects and campaigns."""
    state = StateManager(db_path=str(env["db_path"]))

    async def _seed():
        await state.init_db()
        biz_id = await state.add_business(BusinessProfile(name="SaaS Metrics"))
        prod1_id = await state.add_product(Product(business_id=biz_id, name="MRR Analytics"))
        prod2_id = await state.add_product(Product(business_id=biz_id, name="Churn Predictor"))

        p1 = Prospect(
            first_name="Alice",
            last_name="Smith",
            title="VP Finance",
            email="alice@corp.example",
            product_id=prod1_id,
            status="researched",
        )
        p2 = Prospect(
            first_name="Bob",
            last_name="Jones",
            title="VP Customer Success",
            email="bob@corp.example",
            product_id=prod2_id,
            status="contacted",
        )
        await state.add_prospect(p1)
        await state.add_prospect(p2)
        return prod1_id, prod2_id

    prod1_id, prod2_id = asyncio.run(_seed())

    # Fetch prospects for Product 1
    resp1 = client.get(f"/api/products/{prod1_id}/prospects")
    assert resp1.status_code == 200
    prospects1 = resp1.json()
    assert len(prospects1) == 1
    assert prospects1[0]["email"] == "alice@corp.example"

    # Fetch prospects for Product 2
    resp2 = client.get(f"/api/products/{prod2_id}/prospects")
    assert resp2.status_code == 200
    prospects2 = resp2.json()
    assert len(prospects2) == 1
    assert prospects2[0]["email"] == "bob@corp.example"

    # Deletion guard: Product 1 has prospects, deleting without force should fail
    del_blocked = client.delete(f"/api/products/{prod1_id}")
    assert del_blocked.status_code == 400
    assert "prospect(s) linked" in del_blocked.json()["detail"]

    # Force delete succeeds
    del_forced = client.delete(f"/api/products/{prod1_id}?force=true")
    assert del_forced.status_code == 200


def test_ai_generation_endpoints_api(client):
    """Test local AI generation endpoints for Business Profile and ICP."""
    # Test POST /api/businesses/generate
    mock_biz_data = {
        "name": "Synthetix AI",
        "description": "Synthetix builds enterprise synthetic data pipelines.",
        "website": "https://synthetix.example.com",
        "industry": "Artificial Intelligence",
        "company_size": "20-50",
        "business_location": "London, UK",
        "target_markets": ["Financial Services", "Healthcare"],
        "brand_voice": "Technical, rigorous, confident",
    }
    with patch(
        "sellbird.products_router.generate_business_profile",
        new_callable=AsyncMock,
        return_value=mock_biz_data,
    ) as mock_gen_biz:
        resp = client.post(
            "/api/businesses/generate",
            json={
                "name": "Synthetix AI",
                "website": "https://synthetix.example.com",
                "description": "Enterprise synthetic data",
                "industry": "AI",
                "company_facts": ["Founded in 2023", "HIPAA compliant"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Synthetix AI"
        assert resp.json()["brand_voice"] == "Technical, rigorous, confident"
        assert mock_gen_biz.called

    # Create actual business & product to test ICP generation
    b_resp = client.post("/api/businesses", json={"name": "CyberMesh"})
    biz_id = b_resp.json()["id"]
    p_resp = client.post(
        "/api/products",
        json={"business_id": biz_id, "name": "ZeroTrust Agent", "category": "Security"},
    )
    prod_id = p_resp.json()["id"]

    mock_icp_data = {
        "name": "ZeroTrust Security Leaders",
        "description": "CISOs and Security Directors at mid-market fintechs",
        "target_industries": ["Fintech", "Banking"],
        "company_sizes": ["250-2000"],
        "target_geographies": ["North America"],
        "job_titles": ["CISO", "Director of Information Security"],
        "buyer_personas": ["Security leader balancing audit readiness with developer speed"],
        "pain_points": ["Slow manual compliance reviews", "Fragmented access controls"],
        "buying_signals": ["Preparing for SOC2 Type II audit", "Appointing new CISO"],
        "qualification_criteria": ["AWS or GCP infrastructure", "Over 200 employees"],
    }
    with patch(
        "sellbird.products_router.generate_icp",
        new_callable=AsyncMock,
        return_value=mock_icp_data,
    ) as mock_gen_icp:
        resp = client.post(
            "/api/icps/generate",
            json={"product_id": prod_id, "target_market_hint": "Mid-market fintechs"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "ZeroTrust Security Leaders"
        assert "Fintech" in data["target_industries"]
        assert mock_gen_icp.called
