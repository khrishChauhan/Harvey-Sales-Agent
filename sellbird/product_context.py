"""Product Context resolution and isolation for Harvey Local AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sellbird.models.business import BusinessProfile
from sellbird.models.icp import ICP
from sellbird.models.product import Product
from sellbird.state import StateManager


@dataclass
class ProductContext:
    """Isolated sales context for a specific product, business profile, and ICP."""
    business: BusinessProfile
    product: Product
    icp: ICP

    @property
    def product_id(self) -> str:
        return self.product.id

    @property
    def business_id(self) -> str:
        return self.business.id

    @property
    def product_name(self) -> str:
        return self.product.name

    @property
    def product_description(self) -> str:
        return self.product.description

    @property
    def pricing(self) -> str:
        return self.product.pricing

    @property
    def key_benefits(self) -> list[str]:
        return self.product.key_benefits

    @property
    def objection_responses(self) -> dict[str, str]:
        return self.product.objection_responses

    @property
    def offer(self) -> dict[str, Any]:
        return self.product.offer

    @property
    def target_industries(self) -> list[str]:
        return self.icp.target_industries or self.product.target_industries

    @property
    def job_titles(self) -> list[str]:
        return self.icp.job_titles

    @property
    def buying_signals(self) -> list[str]:
        return self.icp.buying_signals

    @property
    def pain_points(self) -> list[str]:
        return self.icp.pain_points

    def sales_brief(self) -> str:
        """Formatted context block for AI prompts guaranteeing zero cross-product leakage."""
        benefits_list = "\n".join(f"- {b}" for b in self.key_benefits) if self.key_benefits else "- High ROI solutions"
        signals_list = ", ".join(self.buying_signals) if self.buying_signals else "Active in-market signals"
        titles_list = ", ".join(self.job_titles) if self.job_titles else "Key decision makers"

        return (
            f"PRODUCT CONTEXT:\n"
            f"Company: {self.business.name} ({self.business.industry})\n"
            f"Product: {self.product.name}\n"
            f"Description: {self.product.description}\n"
            f"Pricing: {self.product.pricing}\n"
            f"Key Benefits:\n{benefits_list}\n"
            f"Target Roles: {titles_list}\n"
            f"Buying Signals: {signals_list}\n"
        )


async def resolve_product_context(
    state: StateManager,
    product_id: str | None = None,
    config: Any = None,
) -> ProductContext:
    """Resolve the exact ProductContext for a product ID, falling back to default product without leakage."""
    product = None
    business = None
    icp = None

    if product_id:
        product = await state.get_product(product_id)

    # If product not found by ID, get or create default product
    if not product:
        b_id, p_id, i_id = await state.ensure_default_product(config)
        product = await state.get_product(p_id)
        if product:
            business = await state.get_business(product.business_id)
            icp = await state.get_icp(i_id)

    if product and not business:
        business = await state.get_business(product.business_id)

    if product and not icp:
        icps = await state.list_icps(product.id)
        icp = icps[0] if icps else None

    # Fallback safety if DB was completely empty and unseeded
    if not business:
        co_name = getattr(getattr(config, "persona", None), "company", "") or "Harvey Sales Agent"
        business = BusinessProfile(
            id="fallback_business",
            name=co_name,
            description="Autonomous AI Sales Organization",
        )

    if not product:
        prod_cfg = getattr(config, "product", None)
        product = Product(
            id="fallback_product",
            business_id=business.id,
            name=getattr(prod_cfg, "name", "") or "Default Product",
            description=getattr(prod_cfg, "description", "") or "Sales Acceleration Solution",
            pricing=getattr(prod_cfg, "pricing", "") or "Custom",
            key_benefits=getattr(prod_cfg, "key_benefits", []) or [],
            objection_responses=getattr(prod_cfg, "objection_responses", {}) or {},
        )

    if not icp:
        icp_cfg = getattr(config, "icp", None)
        icp = ICP(
            id="fallback_icp",
            product_id=product.id,
            name="Primary ICP",
            target_industries=getattr(icp_cfg, "industries", []) or ["Technology"],
            job_titles=getattr(icp_cfg, "titles", []) or ["VP", "Director", "Founder"],
        )

    return ProductContext(business=business, product=product, icp=icp)
