"""Local AI Product-Specific ICP Generator using Ollama via Brain."""

from __future__ import annotations

import logging
from typing import Any

from sellbird.brain import Brain
from sellbird.models.business import BusinessProfile
from sellbird.models.product import Product

logger = logging.getLogger("sellbird.icp_generator")


async def generate_icp(
    brain: Brain,
    business: BusinessProfile | dict[str, Any],
    product: Product | dict[str, Any],
    target_market_hint: str = "",
) -> dict[str, Any]:
    """Generate a product-specific Ideal Customer Profile (ICP) using local Ollama inference."""
    b_dict = business.model_dump() if hasattr(business, "model_dump") else dict(business)
    p_dict = product.model_dump() if hasattr(product, "model_dump") else dict(product)

    features = p_dict.get("key_features") or []
    benefits = p_dict.get("key_benefits") or []
    usps = p_dict.get("unique_selling_points") or []
    target_ind = p_dict.get("target_industries") or []

    prompt = f"""You are a master Go-To-Market (GTM) strategist and sales architect.
Generate a high-converting Ideal Customer Profile (ICP) tailored specifically for the product below.

COMPANY & PRODUCT CONTEXT:
Company: {b_dict.get('name', 'Our Company')} ({b_dict.get('industry', 'B2B')})
Company Description: {b_dict.get('description', '')}
Product Name: {p_dict.get('name', 'Our Product')}
Product Description: {p_dict.get('description', '')}
Pricing: {p_dict.get('pricing', 'Competitive')}
Key Features: {', '.join(features) if features else 'Core automated workflows'}
Key Benefits: {', '.join(benefits) if benefits else 'Cost efficiency and higher throughput'}
Unique Selling Points: {', '.join(usps) if usps else 'Local execution, zero data leakage'}
Target Markets / Hint: {target_market_hint or ', '.join(target_ind) or 'Global B2B'}

TASK:
Derive the ideal buyer profile that has the highest urgency and willingness to pay for this specific product.

Respond ONLY with a JSON object in exactly this schema:
{{
  "name": "Concise, descriptive ICP title (e.g. Mid-Market B2B SaaS Sales Leaders)",
  "description": "2-3 sentence overview of why this profile has the strongest need and budget for the product",
  "target_industries": ["Industry 1", "Industry 2", "Industry 3"],
  "company_sizes": ["10-50 employees", "50-200 employees"],
  "target_geographies": ["United States", "United Kingdom", "Canada"],
  "job_titles": ["VP Sales", "Head of Revenue Operations", "Chief Commercial Officer"],
  "buyer_personas": ["Revenue Leader scaling outreach without ballooning headcount"],
  "pain_points": [
    "High manual research overhead draining sales rep capacity",
    "Low conversion rates from generic mass outreach"
  ],
  "buying_signals": [
    "Hiring SDRs or BDRs on job boards",
    "Recent funding announcement",
    "Expanding into new geographic territories"
  ],
  "qualification_criteria": [
    "B2B outbound sales model",
    "At least 5 customer-facing reps",
    "Dedicated sales leadership in place"
  ]
}}"""

    result = await brain.think_json(
        prompt,
        agent="icp_strategist",
        task="generate_icp",
    )

    if isinstance(result, dict) and result.get("name"):
        return {
            "product_id": p_dict.get("id", ""),
            "name": str(result.get("name", "Target Decision Makers")),
            "description": str(result.get("description", "")),
            "target_industries": list(result.get("target_industries", ["B2B SaaS"])),
            "company_sizes": list(result.get("company_sizes", ["10-200 employees"])),
            "target_geographies": list(result.get("target_geographies", ["United States"])),
            "job_titles": list(result.get("job_titles", ["Founder", "VP Sales", "Head of Growth"])),
            "buyer_personas": list(result.get("buyer_personas", [])),
            "pain_points": list(result.get("pain_points", [])),
            "buying_signals": list(result.get("buying_signals", [])),
            "qualification_criteria": list(result.get("qualification_criteria", [])),
        }

    # Deterministic fallback baseline
    logger.warning("ICP AI generation returned empty or malformed result; generating structured baseline.")
    default_industries = target_ind if target_ind else ["B2B SaaS", "Technology Services"]
    return {
        "product_id": p_dict.get("id", ""),
        "name": f"{p_dict.get('name', 'Product')} Primary ICP",
        "description": f"Companies with active demand for {p_dict.get('name', 'our product')} and high ROI potential.",
        "target_industries": default_industries,
        "company_sizes": ["10-200 employees"],
        "target_geographies": ["United States", "United Kingdom"],
        "job_titles": ["Founder", "VP Sales", "Head of Business Development", "Chief Revenue Officer"],
        "buyer_personas": ["Executive seeking automated growth and predictable pipeline"],
        "pain_points": [
            "Manual lead qualification is time-consuming",
            "Customer acquisition costs are too high",
        ],
        "buying_signals": [
            "Hiring for commercial and growth roles",
            "Launching new products or regional expansion",
        ],
        "qualification_criteria": [
            "Active outbound sales motions",
            "Decision maker accessible via email or LinkedIn",
        ],
    }
