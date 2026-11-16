"""FastAPI Router for Business Profiles, Products, and ICPs."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
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
