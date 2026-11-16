"""Tests for Phase 3: Local AI Sales Intelligence Engine.

Covers:
- Local embedding generation and cosine similarity
- Semantic prospect-to-ICP matching
- SQLite embedding persistence and similarity retrieval
- Buying-signal classification with local AI and heuristic fallback
- Evidence-based problem statement generation
- Explainable composite prospect scoring
- Prospect research offline gathering
- Enhanced reply classification with product context
- Complete 9-stage offline sales intelligence workflow
- Graceful Ollama-unavailable behavior
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from sellbird.brain import Brain
from sellbird.embeddings import (
    DEFAULT_MODEL_NAME,
    MODEL_LICENCE,
    cosine_similarity,
    embed_text,
    embed_texts,
    prospect_icp_semantic_match,
)
from sellbird.models.business import BusinessProfile
from sellbird.models.icp import ICP
from sellbird.models.product import Product
from sellbird.models.prospect import Prospect
from sellbird.offline_workflow import run_offline_sales_intelligence
from sellbird.problem_statement import generate_problem_statement
from sellbird.product_context import ProductContext
from sellbird.research import gather_company_research
from sellbird.scoring import score_prospect
from sellbird.signals import seed_signal_catalog
from sellbird.signals_classifier import classify_buying_signals
from sellbird.state import StateManager


@pytest.fixture
def state(tmp_path):
    db_file = tmp_path / "test_intel.db"
    mgr = StateManager(db_path=str(db_file))
    asyncio.run(mgr.init_db())
    asyncio.run(seed_signal_catalog(mgr))
    return mgr


@pytest.fixture
def product_context():
    biz = BusinessProfile(name="CyberDefense Inc", industry="Cybersecurity")
    prod = Product(
        business_id=biz.id,
        name="ShieldZero",
        description="Zero-trust access gateway for cloud infrastructure",
        category="Security",
        pricing="$20/user/month",
        key_benefits=["Instant SSO with mTLS", "Zero VPN maintenance"],
        unique_selling_points=["eBPF packet filtering without agent"],
        target_industries=["Fintech", "Healthtech"],
        objection_responses={"Too expensive": "Reduces VPN licensing and ticket costs by 60%."},
        offer={"type": "14-Day Free Pilot", "cta": "Book an onboarding session"},
    )
    icp = ICP(
        product_id=prod.id,
        name="Fintech Security Leaders",
        target_industries=["Fintech", "Banking"],
        job_titles=["Chief Information Security Officer", "CISO", "VP Information Security", "Head of DevSecOps"],
        pain_points=["Legacy VPN outages", "Manual access auditing"],
        buying_signals=["Hiring compliance engineers", "SOC2 Type II prep"],
    )
    return ProductContext(business=biz, product=prod, icp=icp)


# ── 1. Local Embeddings & Cosine Similarity ──

def test_local_embedding_generation_and_cosine_similarity():
    """Verify local embedding dimensions and cosine similarity semantics."""
    assert MODEL_LICENCE == "Apache-2.0"

    v1 = embed_text("Kubernetes container security and zero trust network architecture")
    assert len(v1) == 384
    assert any(x != 0.0 for x in v1)

    # Identical text should have near 1.0 similarity
    sim_same = cosine_similarity(v1, v1)
    assert abs(sim_same - 1.0) < 1e-4

    # Related security text vs completely unrelated gardening text
    v_related = embed_text("Zero trust microsegmentation and container network policy")
    v_unrelated = embed_text("Organic vegetable gardening tips for tomato cultivation")

    sim_related = cosine_similarity(v1, v_related)
    sim_unrelated = cosine_similarity(v1, v_unrelated)

    assert sim_related > sim_unrelated
    assert sim_unrelated < 0.65


# ── 2. Semantic Prospect-to-ICP Matching ──

def test_semantic_prospect_icp_matching():
    """Verify semantic prospect alignment against product ICP."""
    ciso_profile = "Marcus Vance is Chief Information Security Officer (CISO) at FinBank, managing SOC2 compliance and zero-trust cloud security."
    sim, factors = prospect_icp_semantic_match(
        prospect_profile=ciso_profile,
        icp_description="Zero-trust access gateway for cloud infrastructure",
        target_roles=["CISO", "Head of Security"],
        pain_points=["Manual access auditing", "VPN complexity"],
    )
    assert sim >= 0.50
    assert len(factors) > 0
    assert any("CISO" in f for f in factors)


# ── 3. SQLite Embedding Persistence ──

@pytest.mark.asyncio
async def test_embedding_sqlite_persistence(state):
    """Verify storing, retrieving, and similarity querying vectors in SQLite."""
    vec_a = [1.0] + [0.0] * 383
    vec_b = [0.0, 1.0] + [0.0] * 382
    query_vec = [0.0, 0.95] + [0.0] * 382

    emb_id = await state.save_embedding(
        entity_type="prospect",
        entity_id="p101",
        embedding=vec_a,
        product_id="prod_1",
    )
    assert emb_id

    await state.save_embedding(
        entity_type="prospect",
        entity_id="p102",
        embedding=vec_b,
        product_id="prod_1",
    )

    retrieved = await state.get_embedding("prospect", "p101")
    assert retrieved is not None
    assert len(retrieved) == 384
    assert abs(retrieved[0] - 1.0) < 1e-4

    # Similarity search should rank p102 higher than p101 against query_vec
    matches = await state.find_similar_entities("prospect", query_vec, product_id="prod_1", top_k=2)
    assert len(matches) == 2
    assert matches[0][0] == "p102"  # Closest match
    assert matches[0][1] > matches[1][1]


# ── 4. Buying-Signal Classification ──

@pytest.mark.asyncio
async def test_buying_signal_classification(state, product_context):
    """Verify local AI buying signal classification into structured categories."""
    mock_llm_signals = """[
        {
            "signal_type": "HIRING_EXPANSION",
            "title": "Recruiting Cloud Security Architects",
            "relevance": 0.90,
            "confidence": 0.88,
            "evidence": "FinBank is currently hiring 3 Senior Cloud Security Architects for Kubernetes",
            "is_verified": true,
            "source_url": "https://finbank.example/careers",
            "implication": "Expanding cloud footprint directly aligns with ShieldZero"
        },
        {
            "signal_type": "REGULATORY_COMPLIANCE",
            "title": "Preparing for SOC2 Type II Audit",
            "relevance": 0.85,
            "confidence": 0.92,
            "evidence": "Upcoming SOC2 Type II examination scheduled for Q3",
            "is_verified": true,
            "source_url": "https://finbank.example/security",
            "implication": "ShieldZero provides automated audit evidence"
        }
    ]"""

    async def mock_transport(prompt, **kwargs):
        return mock_llm_signals

    brain = Brain(state, transport=mock_transport)
    evidence = [
        {"text": "FinBank is currently hiring 3 Senior Cloud Security Architects for Kubernetes", "source_url": "https://finbank.example/careers"},
        {"text": "Upcoming SOC2 Type II examination scheduled for Q3", "source_url": "https://finbank.example/security"},
    ]

    classified = await classify_buying_signals(brain, product_context, evidence)
    assert len(classified) == 2
    types = [c["signal_type"] for c in classified]
    assert "HIRING_EXPANSION" in types
    assert "REGULATORY_COMPLIANCE" in types
    assert classified[0]["is_verified"] is True

    # Fallback test when Brain raises exception
    failing_brain = Brain(state, transport=AsyncMock(side_effect=RuntimeError("Ollama down")))
    fallback_signals = await classify_buying_signals(failing_brain, product_context, evidence)
    assert len(fallback_signals) >= 1
    assert fallback_signals[0]["signal_type"] == "HIRING_EXPANSION"


# ── 5. Evidence-based Problem Statement Generation ──

@pytest.mark.asyncio
async def test_evidence_based_problem_statement(state, product_context):
    """Verify evidence-grounded problem statement synthesis."""
    mock_problem_json = """{
        "business_challenge": "FinBank is managing complex manual access controls across growing Kubernetes clusters while preparing for SOC2.",
        "supporting_evidence": ["Hiring 3 Cloud Security Architects", "Upcoming SOC2 Type II examination"],
        "product_relevance": "ShieldZero replaces manual access reviews with automated mTLS zero-trust gating.",
        "uncertainties": ["Existing VPN vendor contract expiry date"],
        "summary": "Automating zero-trust container access ahead of upcoming SOC2 audits."
    }"""

    async def mock_problem_transport(prompt, **kwargs):
        return mock_problem_json

    brain = Brain(state, transport=mock_problem_transport)
    p = Prospect(first_name="Marcus", last_name="Vance", title="CISO", company="FinBank")
    facts = ["FinBank is hiring Cloud Security Architects", "Undergoing SOC2 audit"]
    signals = [{"title": "Hiring Security Architects", "evidence": "Careers page posting"}]

    problem = await generate_problem_statement(brain, product_context, p, facts, signals)
    assert "SOC2" in problem.business_challenge
    assert len(problem.supporting_evidence) == 2
    assert "ShieldZero" in problem.product_relevance
    assert len(problem.uncertainties) == 1
    assert "Problem Angle" in problem.writer_note()


# ── 6. Explainable Prospect Scoring ──

def test_explainable_prospect_scoring(product_context):
    """Verify 0-100 score breakdown with positive factors and deductions."""
    p_ideal = Prospect(
        first_name="Alice",
        last_name="CISO",
        title="Chief Information Security Officer",
        company="FinScale",
        industry="Fintech",
        email="alice@finscale.example",
        email_status="verified",
    )
    obs = [{"signal_code": "HIRING_ROLE", "value_text": "Hiring compliance engineers"}]

    res_ideal = score_prospect(
        prospect=p_ideal,
        product_context=product_context,
        company_data={"industry": "Fintech"},
        observations=obs,
        semantic_similarity=0.88,
    )

    assert res_ideal.total_score >= 90
    assert res_ideal.confidence_level == "high"
    assert any("Sector match" in f for f in res_ideal.positive_factors)
    assert any("Target decision-maker" in f for f in res_ideal.positive_factors)
    assert any("Verified email" in f for f in res_ideal.positive_factors)
    assert any("buying signal" in f.lower() for f in res_ideal.positive_factors)

    # Poor match prospect
    p_poor = Prospect(
        first_name="Bob",
        last_name="Gardener",
        title="Head Groundskeeper",
        company="Botanics Co",
        industry="Agriculture",
        email="bob@botanics.example",
        email_status="guess",
    )
    res_poor = score_prospect(
        prospect=p_poor,
        product_context=product_context,
        company_data={"industry": "Agriculture"},
        observations=[],
        semantic_similarity=0.15,
    )

    assert res_poor.total_score <= 35
    assert res_poor.confidence_level == "low"
    assert any("does not match" in d for d in res_poor.negative_factors)


# ── 7. Prospect Research Offline Gathering ──

@pytest.mark.asyncio
async def test_prospect_research_offline_gathering(state, product_context):
    """Verify research gathering works completely offline from SQLite records."""
    cid = "c_defense_1"
    await state.add_observation(
        company_id=cid,
        signal_code="HIRING_ROLE",
        value_text="Security Clearance DevSecOps Engineer",
        evidence_url="https://defensecorp.example/jobs",
    )
    await state.add_observation(
        company_id=cid,
        signal_code="TECH_STACK",
        value_text="Kubernetes on Bare Metal",
        evidence_url="https://defensecorp.example/tech",
    )

    dossier = await gather_company_research(
        state=state,
        company_id=cid,
        domain="defensecorp.example",
        product_id=product_context.product_id,
        offline_only=True,
    )

    assert dossier.offline_cached is True
    assert len(dossier.hiring_roles) == 1
    assert "Security Clearance DevSecOps Engineer" in dossier.hiring_roles[0]
    assert len(dossier.tech_stack) == 1
    assert "Kubernetes on Bare Metal" in dossier.tech_stack[0]
    assert len(dossier.source_urls) == 2


# ── 8. Enhanced Reply Classification ──

@pytest.mark.asyncio
async def test_reply_classification_phase3(state, product_context):
    """Verify reply classification categorizes requests and includes product context."""
    from sellbird.agents.handler import Handler

    captured_prompt = None

    async def mock_llm_classify(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "meeting_requested"

    brain = Brain(state, transport=mock_llm_classify)
    from tests.test_outbox_native import Cfg, Env

    handler = Handler(
        brain=brain,
        state=state,
        config=Cfg(),
        env=Env(),
    )

    p = Prospect(first_name="Marcus", last_name="Vance", title="CISO", company="FinBank", email="m@finbank.example")
    reply = "I'm interested. Can we get on a quick call this Thursday to see a live demo?"

    intent = await handler._classify_intent(reply, p, product_context=product_context)
    assert intent == "meeting_requested"

    # Verify that the LLM prompt received product context
    assert captured_prompt is not None
    assert "ShieldZero" in captured_prompt
    assert "meeting_requested" in captured_prompt
    assert "more_info_requested" in captured_prompt


# ── 9. Complete Offline Sales Intelligence Workflow ──

@pytest.mark.asyncio
async def test_complete_offline_sales_intelligence_workflow(state):
    """Execute all 9 local intelligence stages end-to-end completely offline."""
    async def mock_offline_brain(prompt, **kwargs):
        if "classify this reply" in prompt.lower():
            return "more_info_requested"
        if "write one cold email" in prompt.lower() or "write a short" in prompt.lower():
            return "Marcus — Noticed your team is hiring DevSecOps for air-gapped systems. ShieldZero provides 100% offline container verification."
        if "problem statement" in prompt.lower() or "business challenge" in prompt.lower() or "formulate a realistic" in prompt.lower():
            return """{"business_challenge": "Managing air-gapped container security without outbound network access.", "supporting_evidence": ["Hiring DevSecOps for air-gap"], "product_relevance": "ShieldZero verifies containers offline.", "uncertainties": ["Existing audit timeline"], "summary": "Air-gapped security automation"}"""
        if "buying signals" in prompt.lower():
            return """[{"signal_type": "HIRING_EXPANSION", "title": "Hiring DevSecOps", "relevance": 0.9, "confidence": 0.9, "evidence": "DevSecOps posting", "is_verified": true}]"""
        if "business profile" in prompt.lower() or "synthesize a complete" in prompt.lower():
            return """{"description": "Autonomous security compliance platform", "industry": "Cybersecurity", "target_markets": ["Defense", "Fintech"], "brand_voice": "Technical"}"""
        if "icp" in prompt.lower():
            return """{"name": "Defense Leads", "target_industries": ["Defense"], "job_titles": ["Chief Information Security Officer", "CISO"], "buyer_personas": ["Security architect"], "pain_points": ["Manual audits"], "buying_signals": ["Hiring DevSecOps"], "qualification_criteria": ["Air-gapped servers"]}"""
        return "{}"

    brain = Brain(state, transport=mock_offline_brain)
    report = await run_offline_sales_intelligence(
        state=state,
        brain=brain,
        company_id="comp_test_offline",
        sample_reply="Could you send over technical docs for offline deployments?",
    )

    assert report.is_offline_verified is True
    assert report.business is not None
    assert report.icp is not None
    assert report.research is not None
    assert report.semantic_match is not None
    assert report.problem_statement is not None
    assert report.scoring is not None
    assert report.draft_email is not None
    assert report.reply_classification is not None
    assert report.reply_classification["classified_intent"] == "more_info_requested"


# ── 10. Audit: Score Bounds, Capping, and Anti-Inflation ──

def test_scoring_bounds_capping_and_anti_inflation(product_context):
    """Verify that scores are strictly bounded in [0, 100], capped consistently,
    and semantic similarity never inflates weak/unqualified prospects.
    """
    # Case A: Ideal prospect exceeding 100 raw points must be strictly capped at 100
    p_ideal = Prospect(
        first_name="Alice",
        last_name="CISO",
        title="Chief Information Security Officer",
        company="FinScale",
        industry="Fintech",
        email="alice@finscale.example",
        email_status="verified",
    )
    obs = [{"signal_code": "HIRING_ROLE", "value_text": "Hiring compliance engineers"}]

    res_capped = score_prospect(
        prospect=p_ideal,
        product_context=product_context,
        company_data={"industry": "Fintech"},
        observations=obs,
        semantic_similarity=0.95,  # Raw boost would be 14 pts -> 114 total
    )
    assert res_capped.total_score == 100
    assert sum(res_capped.score_breakdown.values()) == 100

    # Case B: Prospect with missing information must receive 0 points (not 1)
    p_empty = Prospect()
    res_empty = score_prospect(
        prospect=p_empty,
        product_context=product_context,
        company_data={},
        observations=[],
        semantic_similarity=0.0,
    )
    assert res_empty.total_score == 0
    assert res_empty.confidence_level == "low"
    assert any("No industry information" in n for n in res_empty.negative_factors)
    assert any("No job title" in n for n in res_empty.negative_factors)

    # Case C: Weak prospect with high semantic similarity must NOT receive semantic boost
    p_weak = Prospect(
        first_name="Dan",
        last_name="Landscaper",
        title="Botanical Specialist",
        company="Greenhouse Co",
        industry="Horticulture",
        email="dan@greenhouse.example",
        email_status="guess",
    )
    res_weak = score_prospect(
        prospect=p_weak,
        product_context=product_context,
        company_data={"industry": "Horticulture"},
        observations=[],
        semantic_similarity=0.90,  # Unfairly high semantic score from ambiguous text
    )
    assert res_weak.total_score == 5  # Only the pattern guess deliverability points
    assert "semantic_similarity" not in res_weak.score_breakdown
    assert any("Semantic similarity" in n and "not applied" in n for n in res_weak.negative_factors)


# ── 11. Audit: AI Inference vs. Deterministic Fallback Distinction ──

@pytest.mark.asyncio
async def test_ai_vs_deterministic_distinction(state, product_context):
    """Verify that AI-generated intelligence is explicitly distinguishable from heuristic fallbacks."""
    # 1. Buying signal distinction
    ai_json = """[{"signal_type": "HIRING_EXPANSION", "title": "Hiring CISO", "relevance": 0.9, "confidence": 0.9, "evidence": "Careers posting", "is_verified": true}]"""
    async def ai_transport(p, **kw): return ai_json
    ai_brain = Brain(state, transport=ai_transport)
    ev = [{"text": "Hiring CISO for cloud migration", "source_url": "https://example.com"}]

    ai_signals = await classify_buying_signals(ai_brain, product_context, ev)
    assert ai_signals[0]["classification_source"] == "local_ai"

    # Fallback distinction
    failing_brain = Brain(state, transport=AsyncMock(side_effect=RuntimeError("Ollama down")))
    fallback_signals = await classify_buying_signals(failing_brain, product_context, ev)
    assert fallback_signals[0]["classification_source"] == "deterministic_heuristic"

    # 2. Problem statement distinction
    p = Prospect(first_name="John", last_name="Doe", title="CISO", company="FinBank")
    mock_prob_json = """{"business_challenge": "Managing access", "supporting_evidence": ["Careers post"], "product_relevance": "Fits", "uncertainties": ["None"], "summary": "Access issue"}"""
    async def prob_transport(p, **kw): return mock_prob_json
    prob_brain = Brain(state, transport=prob_transport)

    ai_prob = await generate_problem_statement(prob_brain, product_context, p, ["Confirmed fact"])
    assert ai_prob.generation_source == "local_ai"

    fallback_prob = await generate_problem_statement(failing_brain, product_context, p, ["Confirmed fact"])
    assert fallback_prob.generation_source == "deterministic_heuristic"
