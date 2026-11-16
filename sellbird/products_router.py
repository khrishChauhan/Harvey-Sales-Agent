"""FastAPI Router for Business Profiles, Products, and ICPs."""

from __future__ import annotations

import aiosqlite

import logging
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from sellbird.brain import Brain
from sellbird.business_generator import generate_business_profile
from sellbird.config import load_config
from sellbird.icp_generator import generate_icp
from sellbird.models.business import BusinessProfile
from sellbird.models.icp import ICP
from sellbird.models.product import Product
from sellbird.state import StateManager

logger = logging.getLogger("sellbird.products_api")

router = APIRouter(tags=["Multi-Product Architecture"])


def _state() -> StateManager:
    try:
        from sellbird.dashboard import DB_PATH
        return StateManager(db_path=str(DB_PATH))
    except Exception:
        return StateManager()


# ── Request Schemas ──

class BusinessCreateRequest(BaseModel):
    name: str
    description: str = ""
    website: str = ""
    industry: str = ""
    company_size: str = ""
    business_location: str = ""
    target_markets: list[str] = Field(default_factory=list)
    brand_voice: str = ""


class BusinessUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    business_location: Optional[str] = None
    target_markets: Optional[list[str]] = None
    brand_voice: Optional[str] = None


class ProductCreateRequest(BaseModel):
    business_id: str
    name: str
    description: str = ""
    category: str = ""
    pricing: str = ""
    key_features: list[str] = Field(default_factory=list)
    key_benefits: list[str] = Field(default_factory=list)
    unique_selling_points: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    objection_responses: dict[str, str] = Field(default_factory=dict)
    offer: dict[str, Any] = Field(default_factory=dict)


class ProductUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    pricing: Optional[str] = None
    key_features: Optional[list[str]] = None
    key_benefits: Optional[list[str]] = None
    unique_selling_points: Optional[list[str]] = None
    target_industries: Optional[list[str]] = None
    objection_responses: Optional[dict[str, str]] = None
    offer: Optional[dict[str, Any]] = None


class ICPCreateRequest(BaseModel):
    product_id: str
    name: str
    description: str = ""
    target_industries: list[str] = Field(default_factory=list)
    company_sizes: list[str] = Field(default_factory=list)
    target_geographies: list[str] = Field(default_factory=list)
    job_titles: list[str] = Field(default_factory=list)
    buyer_personas: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    buying_signals: list[str] = Field(default_factory=list)
    qualification_criteria: list[str] = Field(default_factory=list)


class ICPUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_industries: Optional[list[str]] = None
    company_sizes: Optional[list[str]] = None
    target_geographies: Optional[list[str]] = None
    job_titles: Optional[list[str]] = None
    buyer_personas: Optional[list[str]] = None
    pain_points: Optional[list[str]] = None
    buying_signals: Optional[list[str]] = None
    qualification_criteria: Optional[list[str]] = None


class GenerateBusinessRequest(BaseModel):
    name: str
    website: str = ""
    description: str = ""
    industry: str = ""
    company_facts: list[str] = Field(default_factory=list)


class GenerateICPRequest(BaseModel):
    product_id: str
    target_market_hint: str = ""


# ── Business Profile Endpoints ──

@router.post("/api/businesses")
async def create_business(req: BusinessCreateRequest):
    """Create a new business profile."""
    state = _state()
    await state.init_db()
    biz = BusinessProfile(**req.model_dump())
    biz_id = await state.add_business(biz)
    saved = await state.get_business(biz_id)
    return saved.model_dump() if saved else {"id": biz_id}


@router.get("/api/businesses")
async def list_businesses():
    """List all business profiles."""
    state = _state()
    await state.init_db()
    businesses = await state.list_businesses()
    return [b.model_dump() for b in businesses]


@router.get("/api/businesses/{business_id}")
async def get_business(business_id: str):
    """Get a specific business profile by ID."""
    state = _state()
    await state.init_db()
    biz = await state.get_business(business_id)
    if not biz:
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' not found")
    return biz.model_dump()


@router.put("/api/businesses/{business_id}")
async def update_business(business_id: str, req: BusinessUpdateRequest):
    """Update a business profile."""
    state = _state()
    await state.init_db()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided for update")
    ok = await state.update_business(business_id, **updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' not found")
    saved = await state.get_business(business_id)
    return saved.model_dump()


@router.delete("/api/businesses/{business_id}")
async def delete_business(business_id: str, force: bool = Query(False)):
    """Delete a business profile (refuses if products are linked unless force=True)."""
    state = _state()
    await state.init_db()
    try:
        ok = await state.delete_business(business_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' not found")
    return {"status": "deleted", "id": business_id}


# ── Product Endpoints ──

@router.post("/api/products")
async def create_product(req: ProductCreateRequest):
    """Create a new product linked to a business profile."""
    state = _state()
    await state.init_db()
    biz = await state.get_business(req.business_id)
    if not biz:
        raise HTTPException(status_code=400, detail=f"Referenced business '{req.business_id}' does not exist")
    prod = Product(**req.model_dump())
    prod_id = await state.add_product(prod)
    saved = await state.get_product(prod_id)
    return saved.model_dump() if saved else {"id": prod_id}

@router.post("/api/businesses/{business_id}/products")
async def create_business_product(business_id: str, payload: dict = Body(...)):
    """Create a new product linked to the path business_id."""
    state = _state()
    await state.init_db()
    biz = await state.get_business(business_id)
    if not biz:
        raise HTTPException(status_code=400, detail=f"Referenced business '{business_id}' does not exist")
    data = dict(payload)
    data["business_id"] = business_id
    prod = Product(**data)
    prod_id = await state.add_product(prod)
    saved = await state.get_product(prod_id)
    return saved.model_dump() if saved else {"id": prod_id}



@router.get("/api/products")
async def list_products(business_id: Optional[str] = Query(None)):
    """List products, optionally filtered by business_id."""
    state = _state()
    await state.init_db()
    products = await state.list_products(business_id=business_id)
    return [p.model_dump() for p in products]


@router.get("/api/products/{product_id}")
async def get_product(product_id: str):
    """Get a product's details."""
    state = _state()
    await state.init_db()
    prod = await state.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    return prod.model_dump()


@router.put("/api/products/{product_id}")
async def update_product(product_id: str, req: ProductUpdateRequest):
    """Update product details."""
    state = _state()
    await state.init_db()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided for update")
    ok = await state.update_product(product_id, **updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    saved = await state.get_product(product_id)
    return saved.model_dump()


@router.delete("/api/products/{product_id}")
async def delete_product(product_id: str, force: bool = Query(False)):
    """Delete a product (refuses if prospects are linked unless force=True)."""
    state = _state()
    await state.init_db()
    try:
        ok = await state.delete_product(product_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    return {"status": "deleted", "id": product_id}


@router.get("/api/products/{product_id}/prospects")
async def get_product_prospects(product_id: str, status: Optional[str] = Query(None)):
    """Get prospects associated with a specific product."""
    state = _state()
    await state.init_db()
    prospects = await state.get_prospects_by_product(product_id, status=status)
    return [p.model_dump() for p in prospects]


@router.get("/api/products/{product_id}/campaigns")
async def get_product_campaigns(product_id: str):
    """Get campaigns associated with a specific product."""
    state = _state()
    await state.init_db()
    campaigns = await state.get_campaigns_by_product(product_id)
    return [c.model_dump() for c in campaigns]


# ── ICP Endpoints ──

@router.post("/api/icps")
async def create_icp(req: ICPCreateRequest):
    """Create an ICP linked to a product."""
    state = _state()
    await state.init_db()
    prod = await state.get_product(req.product_id)
    if not prod:
        raise HTTPException(status_code=400, detail=f"Referenced product '{req.product_id}' does not exist")
    icp = ICP(**req.model_dump())
    icp_id = await state.add_icp(icp)
    saved = await state.get_icp(icp_id)
    return saved.model_dump() if saved else {"id": icp_id}

@router.get("/api/products/{product_id}/icps")
async def get_product_icps(product_id: str):
    """Get ICPs associated with a specific product."""
    state = _state()
    await state.init_db()
    icps = await state.list_icps(product_id=product_id)
    return [i.model_dump() for i in icps]


@router.post("/api/products/{product_id}/icps")
async def create_product_icp(product_id: str, payload: dict = Body(...)):
    """Create an ICP linked to the path product_id."""
    state = _state()
    await state.init_db()
    prod = await state.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=400, detail=f"Referenced product '{product_id}' does not exist")
    data = dict(payload)
    data["product_id"] = product_id
    for list_field in (
        "target_industries", "company_sizes", "target_geographies",
        "job_titles", "buyer_personas", "pain_points",
        "buying_signals", "qualification_criteria",
    ):
        v = data.get(list_field)
        if isinstance(v, str):
            data[list_field] = [v] if v else []
    if "target_titles" in data and "job_titles" not in data:
        data["job_titles"] = data.pop("target_titles")
    if "geographies" in data and "target_geographies" not in data:
        data["target_geographies"] = data.pop("geographies")
    icp = ICP(**data)
    icp_id = await state.add_icp(icp)
    saved = await state.get_icp(icp_id)
    return saved.model_dump() if saved else {"id": icp_id}



@router.get("/api/icps")
async def list_icps(product_id: Optional[str] = Query(None)):
    """List ICPs, optionally filtered by product_id."""
    state = _state()
    await state.init_db()
    icps = await state.list_icps(product_id=product_id)
    return [i.model_dump() for i in icps]


@router.get("/api/icps/{icp_id}")
async def get_icp(icp_id: str):
    """Get ICP details."""
    state = _state()
    await state.init_db()
    icp = await state.get_icp(icp_id)
    if not icp:
        raise HTTPException(status_code=404, detail=f"ICP '{icp_id}' not found")
    return icp.model_dump()


@router.put("/api/icps/{icp_id}")
async def update_icp(icp_id: str, req: ICPUpdateRequest):
    """Update ICP details."""
    state = _state()
    await state.init_db()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided for update")
    ok = await state.update_icp(icp_id, **updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"ICP '{icp_id}' not found")
    saved = await state.get_icp(icp_id)
    return saved.model_dump()


@router.delete("/api/icps/{icp_id}")
async def delete_icp(icp_id: str):
    """Delete an ICP."""
    state = _state()
    await state.init_db()
    ok = await state.delete_icp(icp_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"ICP '{icp_id}' not found")
    return {"status": "deleted", "id": icp_id}


# ── Local AI Generation Endpoints ──

@router.post("/api/businesses/generate")
async def api_generate_business(req: GenerateBusinessRequest):
    """Synthesize a structured business profile using local Ollama inference."""
    state = _state()
    await state.init_db()
    cfg = load_config()
    brain = Brain(state, cfg.brain)
    generated = await generate_business_profile(
        brain=brain,
        business_name=req.name,
        website=req.website,
        description=req.description,
        industry=req.industry,
        company_facts=req.company_facts,
    )
    return generated


@router.post("/api/icps/generate")
async def api_generate_icp(req: GenerateICPRequest):
    """Synthesize a product-specific ICP using local Ollama inference."""
    state = _state()
    await state.init_db()
    product = await state.get_product(req.product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product '{req.product_id}' not found")
    business = await state.get_business(product.business_id)
    if not business:
        business = BusinessProfile(
            name="Our Organization",
            description="B2B Technology Services",
        )

    cfg = load_config()
    brain = Brain(state, cfg.brain)
    generated = await generate_icp(
        brain=brain,
        business=business,
        product=product,
        target_market_hint=req.target_market_hint,
    )
    return generated


# ── Phase 4: Application Integration Endpoints ──

from datetime import datetime, timezone
import httpx
from sellbird.scoring import score_prospect
from sellbird.embeddings import (
    prospect_icp_semantic_match,
    load_embedding_model,
    get_model_cache_dir,
    DEFAULT_MODEL_NAME,
    MODEL_LICENCE,
)
from sellbird.research import gather_company_research
from sellbird.signals_classifier import classify_buying_signals
from sellbird.problem_statement import generate_problem_statement
from sellbird.product_context import resolve_product_context, ProductContext
from sellbird.hardware import detect_hardware


class SetActiveProductRequest(BaseModel):
    product_id: str


class SetActiveICPRequest(BaseModel):
    icp_id: str


class StageEmailRequest(BaseModel):
    subject: str
    body: str
    product_id: Optional[str] = ""
    campaign_id: Optional[str] = ""


@router.get("/api/active-product")
async def get_active_product():
    """Get the currently selected active product."""
    state = _state()
    await state.init_db()
    active_id = await state.get_setting("active_product_id")
    products = await state.list_products()
    if not products:
        return {"active_product_id": "", "product": None}

    selected = None
    if active_id:
        for p in products:
            if p.id == active_id:
                selected = p
                break
    if not selected:
        selected = products[0]
        await state.set_setting("active_product_id", selected.id)

    return {"active_product_id": selected.id, "product": selected.model_dump()}


@router.post("/api/active-product")
async def set_active_product(req: SetActiveProductRequest):
    """Set the active product ID."""
    state = _state()
    await state.init_db()
    prod = await state.get_product(req.product_id)
    if not prod:
        raise HTTPException(status_code=404, detail=f"Product '{req.product_id}' not found")
    await state.set_setting("active_product_id", req.product_id)
    return {"active_product_id": req.product_id, "product": prod.model_dump()}


@router.get("/api/active-icp")
async def get_active_icp(product_id: Optional[str] = Query(None)):
    """Get the active ICP for the active or specified product."""
    state = _state()
    await state.init_db()
    pid = product_id or await state.get_setting("active_product_id")
    active_icp_key = f"active_icp_{pid}" if pid else "active_icp_id"
    active_icp_id = await state.get_setting(active_icp_key)
    icps = await state.list_icps(product_id=pid) if pid else await state.list_icps()
    if not icps:
        return {"active_icp_id": "", "icp": None}

    selected = None
    if active_icp_id:
        for i in icps:
            if i.id == active_icp_id:
                selected = i
                break
    if not selected:
        selected = icps[0]
        await state.set_setting(active_icp_key, selected.id)

    return {"active_icp_id": selected.id, "icp": selected.model_dump()}


@router.post("/api/active-icp")
async def set_active_icp(req: SetActiveICPRequest):
    """Set the active ICP ID."""
    state = _state()
    await state.init_db()
    icp = await state.get_icp(req.icp_id)
    if not icp:
        raise HTTPException(status_code=404, detail=f"ICP '{req.icp_id}' not found")
    pid = icp.product_id
    if pid:
        await state.set_setting(f"active_icp_{pid}", req.icp_id)
    await state.set_setting("active_icp_id", req.icp_id)
    return {"active_icp_id": req.icp_id, "icp": icp.model_dump()}


@router.get("/api/prospects/{prospect_id}/intelligence")
async def get_prospect_intelligence(prospect_id: str, product_id: Optional[str] = Query(None)):
    """Fetch comprehensive local sales intelligence for a prospect."""
    state = _state()
    await state.init_db()
    from sellbird.signals import seed_signal_catalog
    await seed_signal_catalog(state)

    p = await state.get_prospect(prospect_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Prospect '{prospect_id}' not found")

    pid = product_id or p.product_id or await state.get_setting("active_product_id")
    ctx = await resolve_product_context(state, pid)

    company = None
    if p.company_id:
        company = await state.get_company(p.company_id)
    company_data = company.model_dump() if company else {
        "name": p.company or "Company",
        "industry": p.industry or "General B2B",
        "website": "",
    }

    obs = await state.get_observations_for_company(p.company_id) if p.company_id else []

    dossier = await gather_company_research(
        state=state,
        company_id=p.company_id or p.id,
        domain=company_data.get("website", "") or "",
        product_id=ctx.product_id,
        offline_only=True,
    )

    prospect_profile = f"{p.full_name()} is {p.title} at {p.company} in {p.industry or company_data.get('industry', '')}."
    similarity, sim_factors = prospect_icp_semantic_match(
        prospect_profile=prospect_profile,
        icp_description=ctx.product_description,
        target_roles=ctx.job_titles,
        pain_points=ctx.pain_points,
    )

    scoring_result = score_prospect(
        prospect=p,
        product_context=ctx,
        company_data=company_data,
        observations=obs,
        semantic_similarity=similarity,
    )

    evidence = dossier.verified_facts if dossier and dossier.verified_facts else [
        {"text": o.get("value_text") or o.get("signal_code"), "source_url": o.get("evidence_url", "")}
        for o in obs
    ]
    cfg = load_config()
    brain = Brain(state, cfg.brain)
    classified_signals = await classify_buying_signals(brain, ctx, evidence)

    facts = dossier.factual_summary() if dossier else [f"{p.company} operating in {p.industry or 'B2B'}."]
    problem = await generate_problem_statement(
        brain=brain,
        product_context=ctx,
        prospect=p,
        research_facts=facts,
        buying_signals=classified_signals,
    )

    verified_list = []
    if dossier and dossier.verified_facts:
        for f in dossier.verified_facts:
            verified_list.append({
                "content": f.get("content") or f.get("text") or f.get("fact") or "",
                "source_url": f.get("source_url", ""),
                "created_at": f.get("created_at", ""),
            })

    return {
        "success": True,
        "prospect": p.model_dump(),
        "product_name": ctx.product_name,
        "score": scoring_result.total_score,
        "score_breakdown": scoring_result.score_breakdown,
        "semantic_similarity": similarity,
        "similarity_source": "Local SentenceTransformer (all-MiniLM-L6-v2)",
        "verified_facts": verified_list,
        "buying_signals": [
            {**s, "signal_code": s.get("signal_type") or s.get("signal_code", "SIGNAL")}
            for s in classified_signals
        ],
        "problem_statement": problem.summary or problem.business_challenge,
        "company": company_data,
        "product": ctx.product.model_dump() if ctx.product else None,
        "icp": ctx.icp.model_dump() if ctx.icp else None,
        "scoring": scoring_result.to_dict(),
        "semantic_match": {
            "similarity": similarity,
            "explainable_factors": sim_factors,
        },
        "research": dossier.to_dict() if dossier else None,
        "classified_signals": classified_signals,
    }


@router.post("/api/prospects/{prospect_id}/score")
async def api_score_prospect(prospect_id: str, product_id: Optional[str] = Query(None)):
    """Compute and persist an explainable 0-100 score for a prospect."""
    state = _state()
    await state.init_db()
    p = await state.get_prospect(prospect_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Prospect '{prospect_id}' not found")
    pid = product_id or p.product_id or await state.get_setting("active_product_id")
    ctx = await resolve_product_context(state, pid)

    obs = await state.get_observations_for_company(p.company_id) if p.company_id else []
    prospect_profile = f"{p.full_name()} is {p.title} at {p.company} in {p.industry}."
    similarity, _ = prospect_icp_semantic_match(
        prospect_profile=prospect_profile,
        icp_description=ctx.product_description,
        target_roles=ctx.job_titles,
        pain_points=ctx.pain_points,
    )
    res = score_prospect(
        prospect=p,
        product_context=ctx,
        company_data={"industry": p.industry},
        observations=obs,
        semantic_similarity=similarity,
    )
    await state.update_prospect(prospect_id, score=res.total_score)
    return {"success": True, "score": res.total_score, "score_breakdown": res.score_breakdown, "prospect_id": prospect_id, "scoring": res.to_dict()}


@router.post("/api/prospects/{prospect_id}/generate-email")
async def api_generate_email(prospect_id: str, product_id: Optional[str] = Query(None)):
    """Generate a personalized outreach draft without sending."""
    state = _state()
    await state.init_db()
    p = await state.get_prospect(prospect_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Prospect '{prospect_id}' not found")

    pid = product_id or p.product_id or await state.get_setting("active_product_id")
    ctx = await resolve_product_context(state, pid)

    dossier = await gather_company_research(state, p.company_id or p.id, domain="", product_id=ctx.product_id, offline_only=True)
    facts = dossier.factual_summary() if dossier else [f"{p.company} in {p.industry or 'industry'}."]
    cfg = load_config()
    brain = Brain(state, cfg.brain)

    problem = await generate_problem_statement(brain, ctx, p, facts, [])
    problem_hook = problem.writer_note()

    offer_text = ""
    if ctx.product and ctx.product.offer:
        offer_text = ctx.product.offer.get("entry") or ctx.product.offer.get("type", "Free consultation")
    if not offer_text:
        offer_text = "a short demonstration"

    prompt = f"""Write ONE short, personalized B2B cold email.
From Company: {ctx.business_name}
Brand Voice: {ctx.brand_voice or 'Technical and direct'}
Product: {ctx.product_name} ({ctx.product_description})
Offer: {offer_text}
Recipient: {p.full_name()} ({p.title} at {p.company})
Problem Hook: {problem_hook}

Requirements:
- 3 short paragraphs (under 75 words).
- Grounded strictly in the problem hook and offer.
- No flattery or filler."""

    body = await brain.think(prompt, session_id=f"email-draft-{p.id}", agent="writer", task="generate_email")
    if not body:
        benefit = ctx.product.key_benefits[0] if ctx.product and ctx.product.key_benefits else "improving security posture"
        body = f"""Hi {p.first_name},

Noticed {p.company} is scaling its infrastructure and managing {ctx.pain_points[0] if ctx.pain_points else 'operational compliance'}.

{ctx.product_name} provides {benefit} with full local control.

Open to {offer_text} this Thursday?"""

    subject = f"{ctx.product_name} for {p.company}"
    return {
        "success": True,
        "subject": subject,
        "body": body.strip(),
        "product_id": ctx.product_id,
        "product_name": ctx.product_name,
        "prospect_id": p.id,
        "prospect_name": p.full_name(),
        "recipient_email": p.email,
        "problem_summary": problem.summary,
    }


@router.post("/api/prospects/{prospect_id}/stage-email")
async def api_stage_email(prospect_id: str, req: StageEmailRequest):
    """Stage an outreach draft in Outbox awaiting human review."""
    state = _state()
    await state.init_db()
    p = await state.get_prospect(prospect_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Prospect '{prospect_id}' not found")

    outbox_id = await state.add_outbox_item(
        prospect_id=p.id,
        to_email=p.email,
        subject=req.subject,
        body=req.body,
        send_at=datetime.now(timezone.utc).isoformat(),
        status="pending_review",
        campaign_id=req.campaign_id or "",
        product_id=req.product_id or p.product_id or "",
    )
    return {"success": True, "status": "pending_review", "outbox_id": outbox_id, "approval_required": True}


@router.get("/api/analytics")
async def api_analytics(product_id: Optional[str] = Query(None)):
    """Retrieve global and product-specific analytics."""
    state = _state()
    await state.init_db()

    products = await state.list_products()
    businesses = await state.list_businesses()
    # Campaigns
    async with state._connect() as db:
        db.row_factory = aiosqlite.Row
        sql_c = "SELECT * FROM campaigns"
        params_c = []
        if product_id:
            sql_c += " WHERE product_id = ?"
            params_c.append(product_id)
        async with db.execute(sql_c, params_c) as c_c:
            campaigns = [dict(r) for r in await c_c.fetchall()]

    # Query prospects from SQLite
    async with state._connect() as db:
        db.row_factory = aiosqlite.Row
        sql_p = "SELECT * FROM prospects"
        params_p = []
        if product_id:
            sql_p += " WHERE product_id = ?"
            params_p.append(product_id)
        async with db.execute(sql_p, params_p) as c_p:
            prospects_rows = [dict(r) for r in await c_p.fetchall()]

        # Observations
        async with db.execute("SELECT signal_code, COUNT(*) FROM observations GROUP BY signal_code ORDER BY COUNT(*) DESC LIMIT 10") as c_o:
            signal_rows = await c_o.fetchall()
            signal_dist = {r[0]: r[1] for r in signal_rows}

        # Outbox stats
        async with db.execute("SELECT status, COUNT(*) FROM outbox GROUP BY status") as c_out:
            out_rows = await c_out.fetchall()
            outbox_counts = {r[0]: r[1] for r in out_rows}

        # Conversations
        async with db.execute("SELECT intent, COUNT(*) FROM conversations GROUP BY intent") as c_con:
            con_rows = await c_con.fetchall()
            intent_counts = {r[0] or "unclassified": r[1] for r in con_rows}

    filtered_campaigns = campaigns
    qualified = [p for p in prospects_rows if (p.get("score") or 0) >= 60]

    stages = {
        "new": sum(1 for p in prospects_rows if p.get("status") == "new"),
        "contacted": sum(1 for p in prospects_rows if p.get("status") == "contacted"),
        "replied": sum(1 for p in prospects_rows if p.get("status") == "replied"),
        "interested": sum(1 for p in prospects_rows if p.get("status") in ("interested", "accepted_offer", "meeting")),
        "opted_out": sum(1 for p in prospects_rows if p.get("status") == "opted_out"),
    }

    prod_breakdown = []
    for prod in products:
        p_prospects = [p for p in prospects_rows if p.get("product_id") == prod.id] if not product_id else prospects_rows
        p_qual = [p for p in p_prospects if (p.get("score") or 0) >= 60]
        p_camp = [c for c in campaigns if (c.get("product_id") if isinstance(c, dict) else getattr(c, "product_id", "")) == prod.id]
        avg_s = round(sum(p.get("score") or 0 for p in p_prospects) / len(p_prospects), 1) if p_prospects else 0.0
        biz = next((b for b in businesses if b.id == prod.business_id), None)
        prod_breakdown.append({
            "product_id": prod.id,
            "product_name": prod.name,
            "business_name": biz.name if biz else "Business",
            "total_prospects": len(p_prospects),
            "qualified_prospects": len(p_qual),
            "average_score": avg_s,
            "campaigns_count": len(p_camp),
        })

    total_sent = outbox_counts.get("sent", 0)
    total_replies = stages.get("replied", 0) + stages.get("interested", 0)
    conversion_rate = round((stages["interested"] / total_replies * 100), 1) if total_replies else 0.0
    reply_rate = round((total_replies / total_sent * 100), 1) if total_sent else 0.0

    return {
        "success": True,
        "total_prospects": len(prospects_rows),
        "qualified_prospects": len(qualified),
        "staged_emails": outbox_counts.get("pending_review", 0),
        "sent_emails": outbox_counts.get("sent", 0),
        "conversion_rates": {
            "prospect_to_contacted": round((sum(1 for p in prospects_rows if p.get("status") in ("contacted", "replied", "interested")) / len(prospects_rows) * 100), 1) if prospects_rows else 0.0,
            "contacted_to_replied": reply_rate,
            "replied_to_meeting": conversion_rate,
        },
        "global": {
            "total_businesses": len(businesses),
            "total_products": len(products),
            "total_prospects": len(prospects_rows),
            "qualified_prospects": len(qualified),
            "total_campaigns": len(filtered_campaigns),
            "outbox": outbox_counts,
        },
        "by_product": prod_breakdown,
        "product_breakdown": prod_breakdown,
        "pipeline_stages": stages,
        "signal_distribution": signal_dist,
        "buying_signals_distribution": signal_dist,
        "reply_intents": intent_counts,
        "conversion_metrics": {
            "reply_rate_percent": reply_rate,
            "positive_interest_rate_percent": conversion_rate,
        },
    }


@router.get("/api/system/ai-status")
async def api_system_ai_status():
    """Retrieve verified status of local Ollama, embedding model, hardware, and telemetry."""
    state = _state()
    await state.init_db()

    cfg = load_config()
    brain_cfg = getattr(cfg, "brain", None)

    endpoint = getattr(brain_cfg, "endpoint", "http://127.0.0.1:11434") if brain_cfg else "http://127.0.0.1:11434"
    active_model = getattr(brain_cfg, "model", "qwen3:4b") if brain_cfg else "qwen3:4b"
    context_length = getattr(brain_cfg, "context_length", 2048) if brain_cfg else 2048
    temperature = getattr(brain_cfg, "temperature", 0.2) if brain_cfg else 0.2

    ollama_ok = False
    ollama_err = ""
    installed_models = []
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{endpoint}/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                data = resp.json()
                installed_models = [m.get("name") for m in data.get("models", [])]
            else:
                ollama_err = f"Ollama returned HTTP {resp.status_code}"
    except Exception as e:
        ollama_err = f"Could not connect to {endpoint} ({type(e).__name__})"

    cache_dir = get_model_cache_dir()
    emb_cached = (cache_dir / DEFAULT_MODEL_NAME).is_dir()

    hw = detect_hardware()
    hw_dict = hw.to_dict()
    import platform as _plt
    hw_dict["ram_total_gb"] = hw.total_ram_gb
    hw_dict["ram_available_gb"] = hw.available_ram_gb
    hw_dict["gpu_name"] = hw.gpu_model
    hw_dict["gpu_vram_gb"] = round(hw.total_vram_mb / 1024, 1) if hw.total_vram_mb else 0
    hw_dict["platform"] = _plt.system()
    hw_dict["os_release"] = _plt.release()

    logs = await state.get_local_inference_logs(limit=25)
    total_calls = len(logs)
    success_calls = sum(1 for l in logs if l.get("status") == "success")
    failed_calls = sum(1 for l in logs if l.get("status") == "error")
    p_tokens = sum(l.get("prompt_tokens", 0) for l in logs)
    c_tokens = sum(l.get("completion_tokens", 0) for l in logs)
    avg_latency = round(sum(l.get("duration_ms", 0) for l in logs) / total_calls) if total_calls else 0

    return {
        "success": True,
        "ollama_available": ollama_ok,
        "ollama_endpoint": endpoint,
        "active_model": active_model,
        "embedding_model": DEFAULT_MODEL_NAME,
        "embedding_cached": emb_cached,
        "ollama": {
            "available": ollama_ok,
            "endpoint": endpoint,
            "active_model": active_model,
            "context_length": context_length,
            "temperature": temperature,
            "installed_models": installed_models,
            "error_message": ollama_err,
        },
        "embeddings": {
            "available": emb_cached,
            "model_name": DEFAULT_MODEL_NAME,
            "dimension": 384,
            "license": MODEL_LICENCE,
            "cache_path": str(cache_dir / DEFAULT_MODEL_NAME),
            "offline_ready": emb_cached,
        },
        "hardware": hw_dict,
        "telemetry": {
            "total_calls": total_calls,
            "successful_calls": success_calls,
            "failed_calls": failed_calls,
            "total_tokens": p_tokens + c_tokens,
            "total_prompt_tokens": p_tokens,
            "total_completion_tokens": c_tokens,
            "avg_latency_ms": avg_latency,
            "recent_logs": logs[:10],
        },
    }
