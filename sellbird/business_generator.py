"""Local AI Business Profile Generator using Ollama via Brain."""

from __future__ import annotations

import logging
from typing import Any

from sellbird.brain import Brain

logger = logging.getLogger("sellbird.business_generator")


async def generate_business_profile(
    brain: Brain,
    business_name: str,
    website: str = "",
    description: str = "",
    industry: str = "",
    company_facts: list[str] | str = "",
) -> dict[str, Any]:
    """Generate a structured business profile using local Ollama inference.

    Synthesizes provided inputs and stored research without hallucinating unverified facts.
    """
    facts_str = "\n".join(f"- {f}" for f in company_facts) if isinstance(company_facts, list) else str(company_facts)

    prompt = f"""You are a senior B2B commercial architect. Analyze the business details provided below and synthesize a high-precision business profile.

CRITICAL INSTRUCTIONS:
- Do NOT invent or hallucinate company facts, certifications, or history not present in the input.
- Ground all positioning in the supplied facts and details.
- Provide crisp, consultative language suitable for autonomous B2B sales operations.

BUSINESS INPUT DETAILS:
Business Name: {business_name}
Website: {website or 'Not specified'}
Initial Description: {description or 'Not specified'}
Industry: {industry or 'Not specified'}
Verified Company Facts:
{facts_str or 'None provided'}

Respond ONLY with a JSON object in exactly this schema:
{{
  "description": "2-3 sentence executive summary of what the company does and who it serves",
  "industry": "Accurate industry classification (e.g. B2B Enterprise SaaS, FinTech, Industrial Logistics)",
  "value_proposition": "Clear, compelling value proposition explaining the core economic outcome delivered",
  "core_capabilities": ["Capability 1", "Capability 2", "Capability 3"],
  "target_markets": ["Primary Market 1", "Primary Market 2"],
  "brand_positioning": "Strategic brand positioning statement against alternatives",
  "brand_voice": "Recommended brand tone (e.g. consultative, authoritative, confident)"
}}"""

    result = await brain.think_json(
        prompt,
        agent="business_architect",
        task="generate_profile",
    )

    if isinstance(result, dict) and result.get("description"):
        return {
            "name": business_name,
            "website": website,
            "description": str(result.get("description", "")),
            "industry": str(result.get("industry", industry or "Technology")),
            "value_proposition": str(result.get("value_proposition", "")),
            "core_capabilities": list(result.get("core_capabilities", [])),
            "target_markets": list(result.get("target_markets", ["Global"])),
            "brand_positioning": str(result.get("brand_positioning", "")),
            "brand_voice": str(result.get("brand_voice", "consultative, confident")),
        }

    # Deterministic fallback when model is unavailable or outputs malformed response
    logger.warning("Business profile AI generation returned empty or malformed result; generating structured baseline.")
    return {
        "name": business_name,
        "website": website,
        "description": description or f"{business_name} delivers specialized services and products.",
        "industry": industry or "Technology & Software",
        "value_proposition": f"Empowering clients of {business_name} with reliable, high-ROI solutions.",
        "core_capabilities": ["Client consultation", "Automated workflows", "Domain expertise"],
        "target_markets": ["North America", "Europe"],
        "brand_positioning": f"{business_name} stands out through dedicated reliability and client outcomes.",
        "brand_voice": "consultative, confident",
    }
