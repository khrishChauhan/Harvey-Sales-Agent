"""Local AI Buying-Signal Classification for Harvey Local AI Sales Intelligence.

Requirements:
- Classifies signals into:
  * HIRING_EXPANSION
  * COMPANY_GROWTH
  * PRODUCT_LAUNCH
  * TECH_ADOPTION
  * OPERATIONAL_CHALLENGES
  * REGULATORY_COMPLIANCE
- Uses local inference through Ollama (Brain.think_json()).
- Tailored to product ICP and buying signals.
- Distinguishes verified evidence from unverified assumptions.
- Preserves original sources and evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sellbird.brain import Brain
from sellbird.product_context import ProductContext
from sellbird.untrusted import UNTRUSTED_RULE, fence

logger = logging.getLogger("sellbird.signals_classifier")

VALID_SIGNAL_CATEGORIES = {
    "HIRING_EXPANSION",
    "COMPANY_GROWTH",
    "PRODUCT_LAUNCH",
    "TECH_ADOPTION",
    "OPERATIONAL_CHALLENGES",
    "REGULATORY_COMPLIANCE",
}

DEFAULT_SIGNAL_DEFINITIONS = {
    "HIRING_EXPANSION": "Actively recruiting for specific roles, department growth, or headcount expansion",
    "COMPANY_GROWTH": "New office locations, funding rounds, strategic investments, or market expansion",
    "PRODUCT_LAUNCH": "Releasing new products, major features, beta programs, or platform upgrades",
    "TECH_ADOPTION": "Adopting, integrating, or migrating software platforms, infrastructure, or tools",
    "OPERATIONAL_CHALLENGES": "Workflow bottlenecks, negative reviews, system downtime, or scaling pains",
    "REGULATORY_COMPLIANCE": "Preparing for or maintaining compliance standards (e.g. SOC2, HIPAA, GDPR, ISO)",
}


async def classify_buying_signals(
    brain: Brain,
    product_context: ProductContext,
    evidence_items: List[Dict[str, Any]],
    confidence_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Classify factual observations and text evidence into structured buying signals using local AI.

    Parameters:
    - brain: Brain instance running local Ollama inference.
    - product_context: Resolved ProductContext containing product features, benefits, and target ICP.
    - evidence_items: List of dicts with 'text', optional 'source_url', and optional 'date'.
    - confidence_threshold: Minimum confidence required to accept a signal.

    Returns:
    List of classified buying signal dicts.
    """
    if not evidence_items:
        return []

    # Format evidence block
    formatted_evidence = []
    for i, item in enumerate(evidence_items):
        text = str(item.get("text") or item.get("value_text") or "").strip()
        url = str(item.get("source_url") or item.get("evidence_url") or "")
        if text:
            formatted_evidence.append(f"[{i+1}] {text} (Source: {url or 'Company data'})")

    if not formatted_evidence:
        return []

    target_signals_text = ", ".join(product_context.buying_signals) if product_context.buying_signals else "Active business needs"
    icp_roles_text = ", ".join(product_context.job_titles) if product_context.job_titles else "Decision makers"

    prompt = f"""You are a B2B sales intelligence specialist analyzing company research to identify genuine buying signals.

Product being sold:
- Name: {product_context.product_name}
- Description: {product_context.product_description}
- Target Buying Signals: {target_signals_text}
- Target Decision Makers: {icp_roles_text}

Available Evidence gathered from public sources:
{fence(chr(10).join(formatted_evidence), "evidence")}

{UNTRUSTED_RULE}

Instructions:
1. Examine each evidence item for genuine buying signals relevant to our product.
2. Only classify signals supported by the evidence above. Never invent or hallucinate facts.
3. Classify into EXACTLY one of these categories:
   - HIRING_EXPANSION
   - COMPANY_GROWTH
   - PRODUCT_LAUNCH
   - TECH_ADOPTION
   - OPERATIONAL_CHALLENGES
   - REGULATORY_COMPLIANCE
4. Assign relevance (0.0 to 1.0) based on how directly this signal creates demand for {product_context.product_name}.
5. Assign confidence (0.0 to 1.0) based on the strength of the evidence.
6. Set is_verified to true ONLY if backed by direct factual quotes, otherwise false.

Return a JSON array of classified signals (or empty array [] if no relevant signals exist):
[
  {{
    "signal_type": "HIRING_EXPANSION",
    "title": "Hiring Security Engineers",
    "relevance": 0.85,
    "confidence": 0.90,
    "evidence": "Quoted excerpt from evidence",
    "is_verified": true,
    "source_url": "source url if provided",
    "implication": "Why this matters for our pitch"
  }}
]"""

    try:
        result = await brain.think_json(
            prompt,
            session_id="sellbird-signal-classification",
            agent="scout",
            task="classify_buying_signals",
        )
    except Exception as e:
        logger.warning(f"Local AI buying-signal classification failed: {e}")
        return _heuristic_signal_classification(product_context, evidence_items)

    classified = []
    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue
            cat = str(item.get("signal_type") or "").upper().strip()
            if cat not in VALID_SIGNAL_CATEGORIES:
                continue
            conf = float(item.get("confidence") or 0.0)
            if conf < confidence_threshold:
                continue

            classified.append({
                "signal_type": cat,
                "title": str(item.get("title") or cat).strip()[:150],
                "relevance": round(float(item.get("relevance") or 0.5), 2),
                "confidence": round(conf, 2),
                "evidence": str(item.get("evidence") or "").strip()[:500],
                "is_verified": bool(item.get("is_verified", False)),
                "source_url": str(item.get("source_url") or "").strip(),
                "implication": str(item.get("implication") or "").strip()[:300],
            })

    if not classified:
        return _heuristic_signal_classification(product_context, evidence_items)

    return classified


def _heuristic_signal_classification(
    product_context: ProductContext,
    evidence_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deterministic keyword-based fallback if Ollama is unavailable."""
    signals = []
    keywords_map = {
        "HIRING_EXPANSION": ["hiring", "careers", "open role", "recruiting", "joined as", "new lead"],
        "COMPANY_GROWTH": ["raised", "series a", "series b", "seed", "expansion", "new office", "growth"],
        "PRODUCT_LAUNCH": ["launched", "releasing", "announced", "feature", "v2", "beta", "new platform"],
        "TECH_ADOPTION": ["migrating", "stack", "kubernetes", "aws", "snowflake", "salesforce", "datadog"],
        "OPERATIONAL_CHALLENGES": ["downtime", "slow", "incident", "legacy", "migration pain", "struggling"],
        "REGULATORY_COMPLIANCE": ["soc2", "hipaa", "gdpr", "iso27001", "audit", "compliance", "pci-dss"],
    }

    for item in evidence_items:
        text = str(item.get("text") or item.get("value_text") or "")
        url = str(item.get("source_url") or item.get("evidence_url") or "")
        low = text.lower()
        for cat, kw_list in keywords_map.items():
            hit = next((kw for kw in kw_list if kw in low), None)
            if hit:
                signals.append({
                    "signal_type": cat,
                    "title": f"Detected {cat.replace('_', ' ').title()}",
                    "relevance": 0.70,
                    "confidence": 0.75,
                    "evidence": text[:300],
                    "is_verified": bool(url),
                    "source_url": url,
                    "implication": f"Identified trigger keyword '{hit}' relevant to sales workflow",
                })
                break

    return signals
