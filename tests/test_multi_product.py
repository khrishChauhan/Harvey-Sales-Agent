"""Tests for Phase 2: Multi-Product Sales Architecture."""

import os
import tempfile
import pytest
import pytest_asyncio

from sellbird.brain import Brain
from sellbird.business_generator import generate_business_profile
from sellbird.icp_generator import generate_icp
from sellbird.models.business import BusinessProfile
from sellbird.models.campaign import Campaign, EmailStep
from sellbird.models.icp import ICP
from sellbird.models.product import Product
from sellbird.models.prospect import Prospect
from sellbird.product_context import resolve_product_context
from sellbird.state import StateManager


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_multi_product.db")
        sm = StateManager(db_path)
        await sm.init_db()
        yield sm


# ── 1. Creating a Business ──

@pytest.mark.asyncio
async def test_create_business(state):
    biz = BusinessProfile(
        name="Acme Enterprise",
        description="Global B2B Supply Chain Software",
        website="https://acme.example",
        industry="Logistics Tech",
        company_size="500-1000",
        business_location="Chicago, IL",
        target_markets=["North America", "Europe"],
        brand_voice="authoritative, consultative",
    )
    b_id = await state.add_business(biz)
    assert b_id, "Business ID must be created"

    fetched = await state.get_business(b_id)
    assert fetched is not None
    assert fetched.name == "Acme Enterprise"
    assert fetched.industry == "Logistics Tech"
    assert fetched.target_markets == ["North America", "Europe"]


# ── 2. Creating Multiple Products & 7. Business-Product Relationships ──

@pytest.mark.asyncio
async def test_create_multiple_products(state):
    b_id = await state.add_business(BusinessProfile(name="Nexus Holdings"))

    p1 = Product(
        business_id=b_id,
        name="Nexus Freight",
        description="Automated route optimization",
        pricing="$1,200/mo",
        key_features=["Dynamic dispatch", "Fuel tracking"],
        key_benefits=["Cut fuel costs by 18%"],
        target_industries=["Trucking", "Freight Forwarding"],
    )
    p2 = Product(
        business_id=b_id,
        name="Nexus Warehouse",
        description="Robotic inventory management",
        pricing="$2,500/mo",
        key_features=["Barcode scanning", "Pallet tracking"],
        key_benefits=["99.9% inventory accuracy"],
        target_industries=["3PL", "Warehousing"],
    )

    p1_id = await state.add_product(p1)
    p2_id = await state.add_product(p2)
    assert p1_id != p2_id

    # List products for business
    products = await state.list_products(business_id=b_id)
    assert len(products) == 2
    names = {p.name for p in products}
    assert names == {"Nexus Freight", "Nexus Warehouse"}


# ── 3. Creating Multiple ICPs for One Product ──

@pytest.mark.asyncio
async def test_create_multiple_icps_for_one_product(state):
    b_id = await state.add_business(BusinessProfile(name="CloudMetrics"))
    p_id = await state.add_product(Product(business_id=b_id, name="LogAnalyzer"))

    icp1 = ICP(
        product_id=p_id,
        name="Fintech DevOps Leads",
        description="DevOps managers at Series B+ Fintechs",
        target_industries=["Fintech", "Banking"],
        company_sizes=["50-200"],
        job_titles=["DevOps Manager", "VP Infrastructure"],
        pain_points=["Slow incident resolution"],
    )
    icp2 = ICP(
        product_id=p_id,
        name="HealthTech Compliance Officers",
        description="Security officers needing audit logs",
        target_industries=["HealthTech", "Healthcare"],
        company_sizes=["100-500"],
        job_titles=["CISO", "Compliance Director"],
        pain_points=["HIPAA audit readiness"],
    )

    id1 = await state.add_icp(icp1)
    id2 = await state.add_icp(icp2)
    assert id1 != id2

    icps = await state.list_icps(product_id=p_id)
    assert len(icps) == 2
    titles = {i.name for i in icps}
    assert titles == {"Fintech DevOps Leads", "HealthTech Compliance Officers"}


# ── 4. Product Retrieval & 5. Product Updates ──

@pytest.mark.asyncio
async def test_product_retrieval_and_updates(state):
    b_id = await state.add_business(BusinessProfile(name="DevCorp"))
    p_id = await state.add_product(Product(
        business_id=b_id,
        name="DevAudit",
        pricing="$300/mo",
        key_features=["Static Analysis"],
    ))

    # Retrieval
    prod = await state.get_product(p_id)
    assert prod is not None
    assert prod.name == "DevAudit"
    assert prod.pricing == "$300/mo"

    # Update
    ok = await state.update_product(
        p_id,
        pricing="$450/mo",
        key_features=["Static Analysis", "Dependency Scanning"],
    )
    assert ok is True

    updated = await state.get_product(p_id)
    assert updated.pricing == "$450/mo"
    assert updated.key_features == ["Static Analysis", "Dependency Scanning"]


# ── 6. Product Deletion Behaviour & Business Relationships ──

@pytest.mark.asyncio
async def test_product_and_business_deletion_safeguards(state):
    b_id = await state.add_business(BusinessProfile(name="SafeDelete Corp"))
    p_id = await state.add_product(Product(business_id=b_id, name="Test Product"))

    # Add a prospect linked to this product
    await state.add_prospect(Prospect(
        id="prospect-del-1",
        product_id=p_id,
        first_name="Alice",
        last_name="Smith",
        email="alice@smith.example",
        title="CEO",
    ))

    # Attempting to delete product with linked prospects must fail safely
    with pytest.raises(ValueError, match="Cannot delete product"):
        await state.delete_product(p_id, force=False)

    # Deleting business with linked products must fail safely
    with pytest.raises(ValueError, match="Cannot delete business"):
        await state.delete_business(b_id, force=False)

    # Deletion with force=True succeeds
    ok = await state.delete_product(p_id, force=True)
    assert ok is True
    assert await state.get_product(p_id) is None


# ── 8. Product-Specific Prospect Filtering ──

@pytest.mark.asyncio
async def test_product_specific_prospect_filtering(state):
    b_id = await state.add_business(BusinessProfile(name="MultiBiz"))
    p1 = await state.add_product(Product(business_id=b_id, name="Product A"))
    p2 = await state.add_product(Product(business_id=b_id, name="Product B"))

    await state.add_prospect(Prospect(
        id="pa1", product_id=p1, first_name="A1", last_name="User",
        email="a1@test.example", title="CTO", status="new",
    ))
    await state.add_prospect(Prospect(
        id="pa2", product_id=p1, first_name="A2", last_name="User",
        email="a2@test.example", title="VP", status="contacted",
    ))
    await state.add_prospect(Prospect(
        id="pb1", product_id=p2, first_name="B1", last_name="User",
        email="b1@test.example", title="Director", status="new",
    ))

    p1_prospects = await state.get_prospects_by_product(p1)
    assert len(p1_prospects) == 2
    assert {p.id for p in p1_prospects} == {"pa1", "pa2"}

    p1_new_only = await state.get_prospects_by_product(p1, status="new")
    assert len(p1_new_only) == 1
    assert p1_new_only[0].id == "pa1"

    p2_prospects = await state.get_prospects_by_product(p2)
    assert len(p2_prospects) == 1
    assert p2_prospects[0].id == "pb1"


# ── 9. Product-Specific Campaign Association ──

@pytest.mark.asyncio
async def test_product_specific_campaign_association(state):
    b_id = await state.add_business(BusinessProfile(name="CampaignCorp"))
    p1 = await state.add_product(Product(business_id=b_id, name="Security Suite"))
    p2 = await state.add_product(Product(business_id=b_id, name="Billing Suite"))

    c1 = Campaign(
        id="camp-1", product_id=p1, name="Security Q1 Outreach",
        sequence=[EmailStep(step=1, subject="Security alert", body="Check your posture")],
    )
    c2 = Campaign(
        id="camp-2", product_id=p2, name="Billing Q1 Outreach",
        sequence=[EmailStep(step=1, subject="Cut billing fees", body="Automate invoices")],
    )

    await state.add_campaign(c1)
    await state.add_campaign(c2)

    p1_campaigns = await state.get_campaigns_by_product(p1)
    assert len(p1_campaigns) == 1
    assert p1_campaigns[0].name == "Security Q1 Outreach"

    p2_campaigns = await state.get_campaigns_by_product(p2)
    assert len(p2_campaigns) == 1
    assert p2_campaigns[0].name == "Billing Q1 Outreach"


# ── 10. Database Migration v18 & 11. Existing Single-Product Compatibility ──

@pytest.mark.asyncio
async def test_database_migration_and_single_product_backfill(state):
    # Seed an unassociated prospect (legacy record without product_id)
    await state.add_prospect(Prospect(
        id="legacy-prospect",
        first_name="Old",
        last_name="Contact",
        email="legacy@test.example",
        title="Founder",
    ))

    # ensure_default_product backfills unassigned records
    b_id, p_id, i_id = await state.ensure_default_product()
    assert b_id and p_id and i_id

    # Verify the legacy record now has the default product_id
    prospect = await state.get_prospect("legacy-prospect")
    assert prospect.product_id == p_id

    # Verify default product and ICP were seeded
    default_prod = await state.get_product(p_id)
    assert default_prod is not None
    assert default_prod.business_id == b_id

    default_icp = await state.get_icp(i_id)
    assert default_icp is not None
    assert default_icp.product_id == p_id


# ── 12. AI Business Profile Generation through Brain ──

@pytest.mark.asyncio
async def test_ai_business_profile_generation(state):
    # Mock transport returning structured profile JSON
    async def mock_transport(prompt, **kwargs):
        return """{
            "description": "Synthesized AI Logistics platform delivering real-time route dispatch.",
            "industry": "B2B Supply Chain & Logistics",
            "value_proposition": "Reduces empty freight miles by 24% through automated load matching.",
            "core_capabilities": ["Automated Dispatch", "Telematics Integration", "Dynamic Pricing"],
            "target_markets": ["North America", "UK"],
            "brand_positioning": "The enterprise standard for intelligent freight orchestration.",
            "brand_voice": "authoritative, consultative"
        }"""

    brain = Brain(state, transport=mock_transport)
    profile = await generate_business_profile(
        brain=brain,
        business_name="DispatchAI",
        website="https://dispatchai.example",
        description="Route optimization software",
        industry="Logistics",
        company_facts=["Founded in 2022", "Processes 50k loads monthly"],
    )

    assert profile["name"] == "DispatchAI"
    assert profile["industry"] == "B2B Supply Chain & Logistics"
    assert "Reduces empty freight miles" in profile["value_proposition"]
    assert len(profile["core_capabilities"]) == 3


# ── 13. AI ICP Generation through Brain ──

@pytest.mark.asyncio
async def test_ai_icp_generation(state):
    async def mock_transport(prompt, **kwargs):
        return """{
            "name": "Mid-Market Freight Dispatch Leaders",
            "description": "Logistics companies with 20-100 trucks seeking dispatch automation.",
            "target_industries": ["Freight Transport", "3PL Logistics"],
            "company_sizes": ["20-100 employees"],
            "target_geographies": ["United States"],
            "job_titles": ["VP Dispatch", "Director of Logistics", "Operations Lead"],
            "buyer_personas": ["Operations executive striving to eliminate manual phone dispatch"],
            "pain_points": ["Manual dispatcher bottlenecks", "Unpredictable deadhead miles"],
            "buying_signals": ["Hiring 5+ drivers on Indeed", "Added 10 new trailers"],
            "qualification_criteria": ["Operates fleet of at least 15 commercial vehicles"]
        }"""

    brain = Brain(state, transport=mock_transport)
    business = BusinessProfile(id="b1", name="DispatchAI", industry="Logistics")
    product = Product(id="p1", business_id="b1", name="SmartDispatcher", description="Automated truck dispatch")

    icp = await generate_icp(brain=brain, business=business, product=product)

    assert icp["product_id"] == "p1"
    assert icp["name"] == "Mid-Market Freight Dispatch Leaders"
    assert "Freight Transport" in icp["target_industries"]
    assert "VP Dispatch" in icp["job_titles"]
    assert len(icp["buying_signals"]) == 2


# ── 14. No Cross-Product Context Leakage ──

@pytest.mark.asyncio
async def test_no_cross_product_context_leakage(state):
    b1_id = await state.add_business(BusinessProfile(name="CyberShield Inc"))
    p1_id = await state.add_product(Product(
        business_id=b1_id,
        name="FirewallPro",
        description="Next-generation cloud firewall for financial institutions",
        pricing="$3,000/mo",
        key_benefits=["Zero packet loss", "SOC2 Type II compliant"],
    ))
    await state.add_icp(ICP(
        product_id=p1_id,
        name="Bank CISOs",
        job_titles=["Chief Information Security Officer"],
    ))

    b2_id = await state.add_business(BusinessProfile(name="HealthCare Soft"))
    p2_id = await state.add_product(Product(
        business_id=b2_id,
        name="MedScheduler",
        description="Patient appointment scheduling for dental clinics",
        pricing="$150/mo",
        key_benefits=["Reduce no-shows by 40%"],
    ))
    await state.add_icp(ICP(
        product_id=p2_id,
        name="Clinic Office Managers",
        job_titles=["Dental Office Manager"],
    ))

    # Resolve context for Product 1
    ctx1 = await resolve_product_context(state, p1_id)
    brief1 = ctx1.sales_brief()

    # Resolve context for Product 2
    ctx2 = await resolve_product_context(state, p2_id)
    brief2 = ctx2.sales_brief()

    # Verify complete isolation: Product 1 details must NOT appear in Product 2 and vice versa
    assert "FirewallPro" in brief1
    assert "SOC2 Type II" in brief1
    assert "Chief Information Security Officer" in brief1
    assert "MedScheduler" not in brief1
    assert "dental" not in brief1

    assert "MedScheduler" in brief2
    assert "Reduce no-shows by 40%" in brief2
    assert "Dental Office Manager" in brief2
    assert "FirewallPro" not in brief2
    assert "SOC2" not in brief2
