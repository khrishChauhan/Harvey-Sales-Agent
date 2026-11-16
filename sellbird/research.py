"""Local Prospect and Company Research Engine for Harvey Local AI Sales Intelligence.

Requirements:
- Reuses existing collectors and local data.
- Uses BeautifulSoup / local scraping on permitted public sites.
- Extracts company description, products/services, hiring activity, tech stack.
- Distinguishes verified factual evidence from AI interpretations.
- Saves source URLs, research timestamps, and product_id.
- Operates 100% offline using stored SQLite observations when network is unavailable.
- Never invents facts when information is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from sellbird.brain import Brain
from sellbird.product_context import ProductContext
from sellbird.state import StateManager
from sellbird.untrusted import UNTRUSTED_RULE, fence

logger = logging.getLogger("sellbird.research")


@dataclass
class ResearchDossier:
    """Consolidated research profile for a target account."""
    company_id: str
    domain: str
    product_id: str
    verified_facts: List[Dict[str, str]] = field(default_factory=list)
    ai_interpretations: List[Dict[str, Any]] = field(default_factory=list)
    hiring_roles: List[str] = field(default_factory=list)
    tech_stack: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    research_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    offline_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "domain": self.domain,
            "product_id": self.product_id,
            "verified_facts": self.verified_facts,
            "ai_interpretations": self.ai_interpretations,
            "hiring_roles": self.hiring_roles,
            "tech_stack": self.tech_stack,
            "source_urls": self.source_urls,
            "research_date": self.research_date,
            "offline_cached": self.offline_cached,
        }

    def factual_summary(self) -> List[str]:
        """Return clean list of verified factual statements for prompt context."""
        facts = []
        for f in self.verified_facts:
            facts.append(f"{f.get('fact', '')} (Source: {f.get('source_url', 'public site')})")
        if self.hiring_roles:
            facts.append(f"Actively hiring: {', '.join(self.hiring_roles[:3])}")
        if self.tech_stack:
            facts.append(f"Verified tech on site: {', '.join(self.tech_stack[:5])}")
        return facts or ["Public business profile on file."]


async def gather_company_research(
    state: StateManager,
    company_id: str,
    domain: str,
    product_id: str = "",
    brain: Optional[Brain] = None,
    product_context: Optional[ProductContext] = None,
    html_content: Optional[str] = None,
    offline_only: bool = False,
) -> ResearchDossier:
    """Gather comprehensive research from local database records and public pages."""
    domain_clean = (domain or "").lower().strip()
    dossier = ResearchDossier(
        company_id=company_id,
        domain=domain_clean,
        product_id=product_id,
    )

    # 1. Pull existing verified observations from SQLite
    obs_list = []
    try:
        obs_list = await state.get_observations(company_id=company_id)
    except Exception as e:
        logger.debug(f"Could not load observations for {company_id}: {e}")

    for o in obs_list:
        code = o.get("signal_code")
        val = o.get("value_text") or str(o.get("value_num") or "")
        url = o.get("evidence_url") or f"https://{domain_clean}"

        if url and url not in dossier.source_urls:
            dossier.source_urls.append(url)

        if code == "HIRING_ROLE" and val:
            if val not in dossier.hiring_roles:
                dossier.hiring_roles.append(val)
            dossier.verified_facts.append({
                "category": "hiring",
                "fact": f"Company posted open role: {val}",
                "source_url": url,
            })
        elif code == "TECH_STACK" and val:
            if val not in dossier.tech_stack:
                dossier.tech_stack.append(val)
            dossier.verified_facts.append({
                "category": "technology",
                "fact": f"Technology identified: {val}",
                "source_url": url,
            })
        elif val:
            dossier.verified_facts.append({
                "category": "observation",
                "fact": f"{code}: {val}",
                "source_url": url,
            })

    # 2. Extract facts from HTML if provided and not offline_only
    if html_content and not offline_only:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            # Extract title and meta description
            title_tag = soup.find("title")
            if title_tag and title_tag.string:
                title_str = title_tag.string.strip()
                dossier.verified_facts.append({
                    "category": "metadata",
                    "fact": f"Website title: {title_str[:120]}",
                    "source_url": f"https://{domain_clean}",
                })

            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if desc_tag and desc_tag.get("content"):
                desc_str = desc_tag["content"].strip()
                dossier.verified_facts.append({
                    "category": "metadata",
                    "fact": f"Site description: {desc_str[:250]}",
                    "source_url": f"https://{domain_clean}",
                })

            # Check headings
            for h in soup.find_all(["h1", "h2"], limit=4):
                text = h.get_text(strip=True)
                if text and len(text) > 15:
                    dossier.verified_facts.append({
                        "category": "heading",
                        "fact": f"Public header: {text[:150]}",
                        "source_url": f"https://{domain_clean}",
                    })
        except Exception as e:
            logger.debug(f"HTML parsing failed for {domain_clean}: {e}")
    else:
        dossier.offline_cached = True

    # 3. Local AI Interpretation (Strictly distinguished from verified facts)
    if brain is not None and dossier.verified_facts and product_context is not None:
        try:
            ai_notes = await _generate_ai_interpretations(brain, product_context, dossier.verified_facts)
            dossier.ai_interpretations = ai_notes
        except Exception as e:
            logger.debug(f"AI interpretation failed: {e}")

    return dossier


async def _generate_ai_interpretations(
    brain: Brain,
    product_context: ProductContext,
    verified_facts: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Generate strictly marked AI inferences from verified factual evidence."""
    facts_lines = [f"- {f.get('fact')} ({f.get('source_url')})" for f in verified_facts[:8]]

    prompt = f"""You are analyzing verified company facts to form sales hypotheses.

TARGET PRODUCT:
- {product_context.product_name}: {product_context.product_description}

VERIFIED COMPANY FACTS (ONLY use what is listed here):
{fence(chr(10).join(facts_lines), "verified facts")}

{UNTRUSTED_RULE}

Instructions:
1. Synthesize 1-2 realistic business hypotheses about this company based ONLY on the verified facts above.
2. Label these as AI interpretations, not confirmed facts.
3. If facts are sparse, state: "Company is actively operating in public domain; operational specifics require discovery."

Return a JSON array:
[
  {{
    "topic": "Operational Focus",
    "interpretation": "Interpretation text grounded in facts",
    "confidence": 0.80
  }}
]"""

    try:
        data = await brain.think_json(
            prompt,
            session_id="research-ai-interpretation",
            agent="scout",
            task="interpret_research",
        )
        if isinstance(data, list):
            return [
                {
                    "topic": str(item.get("topic") or "Sales Angle").strip(),
                    "interpretation": str(item.get("interpretation") or "").strip(),
                    "confidence": round(float(item.get("confidence") or 0.6), 2),
                }
                for item in data if isinstance(item, dict) and item.get("interpretation")
            ]
    except Exception as e:
        logger.debug(f"think_json interpretation failed: {e}")

    return []
