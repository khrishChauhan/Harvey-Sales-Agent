"""Evidence-based Problem Statement Generation for Harvey Local AI Sales Intelligence.

Requirements:
- Generates prospect-specific business challenge analysis.
- Explains:
  1. What business challenge the prospect may be facing.
  2. Which available evidence supports that conclusion.
  3. Why the selected product is relevant.
  4. What remains uncertain or requires verification.
- Strictly grounded in verified research evidence. Never invents facts.
- Output directly feeds the Writer workflow for personalized, problem-led outreach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sellbird.brain import Brain
from sellbird.models.prospect import Prospect
from sellbird.product_context import ProductContext
from sellbird.untrusted import UNTRUSTED_RULE, fence

logger = logging.getLogger("sellbird.problem_statement")


@dataclass
class ProblemStatement:
    """Structured, evidence-grounded problem statement for a prospect."""
    business_challenge: str
    supporting_evidence: List[str] = field(default_factory=list)
    product_relevance: str = ""
    uncertainties: List[str] = field(default_factory=list)
    summary: str = ""
    generation_source: str = "local_ai"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_challenge": self.business_challenge,
            "supporting_evidence": self.supporting_evidence,
            "product_relevance": self.product_relevance,
            "uncertainties": self.uncertainties,
            "summary": self.summary,
            "generation_source": self.generation_source,
        }

    def writer_note(self) -> str:
        """Condensed synthesis for injection into cold email prompts."""
        if not self.business_challenge:
            return ""
        evidence_str = " | ".join(self.supporting_evidence[:2]) if self.supporting_evidence else "public profile"
        return (
            f"Core Problem Angle: {self.business_challenge}\n"
            f"Observed Evidence: {evidence_str}\n"
            f"Why We Fit: {self.product_relevance}"
        )


async def generate_problem_statement(
    brain: Brain,
    product_context: ProductContext,
    prospect: Prospect,
    research_facts: List[str],
    buying_signals: Optional[List[Dict[str, Any]]] = None,
) -> ProblemStatement:
    """Synthesize an evidence-backed problem statement using local Ollama inference.

    Strict rules:
    - If evidence is weak or absent, explicitly states what is uncertain.
    - Never invents problems that cannot be traced to provided research facts or signals.
    """
    signals = buying_signals or []
    signals_text = []
    for s in signals:
        title = s.get("title") or s.get("signal_type")
        ev = s.get("evidence") or ""
        signals_text.append(f"- {title}: {ev}")

    signals_block = "\n".join(signals_text) if signals_text else "None detected yet."
    facts_block = "\n".join(f"- {f}" for f in research_facts) if research_facts else "- Standard public company listing."

    prompt = f"""You are a consultative sales strategist. Formulate a realistic, evidence-based problem statement for this prospect.

OUR SOLUTION:
- Product: {product_context.product_name}
- What it solves: {product_context.product_description}
- Key Benefits: {', '.join(product_context.key_benefits)}
- Target Pain Points: {', '.join(product_context.pain_points)}

THE PROSPECT:
- Name: {prospect.full_name()}
- Title: {prospect.title}
- Company: {prospect.company} ({prospect.industry or 'industry not specified'})

CONFIRMED RESEARCH FACTS:
{fence(facts_block, "confirmed facts")}

VERIFIED BUYING SIGNALS:
{fence(signals_block, "buying signals")}

{UNTRUSTED_RULE}

INSTRUCTIONS:
1. Identify the most probable operational or strategic challenge the prospect faces in their role.
2. Ground your analysis strictly in the verified facts and buying signals above.
3. If facts are minimal, do NOT invent problems — describe a typical challenge for this title and explicitly list what is uncertain.
4. Explain why our product is specifically relevant to resolving this challenge.

Return ONLY a JSON object:
{{
  "business_challenge": "Concise explanation of the challenge (1-2 sentences)",
  "supporting_evidence": ["Fact 1 supporting this", "Fact 2 supporting this"],
  "product_relevance": "Why our product specifically helps resolve this (1 sentence)",
  "uncertainties": ["What remains unconfirmed or requires discovery question"],
  "summary": "Punchy one-sentence problem hook for cold outreach"
}}"""

    try:
        data = await brain.think_json(
            prompt,
            session_id=f"problem-statement-{prospect.id}",
            agent="writer",
            task="generate_problem_statement",
        )
    except Exception as e:
        logger.warning(f"Local AI problem statement generation failed: {e}")
        data = None

    if isinstance(data, dict) and data.get("business_challenge"):
        return ProblemStatement(
            business_challenge=str(data.get("business_challenge", "")).strip(),
            supporting_evidence=list(data.get("supporting_evidence", [])),
            product_relevance=str(data.get("product_relevance", "")).strip(),
            uncertainties=list(data.get("uncertainties", [])),
            summary=str(data.get("summary", "")).strip(),
        )

    # Deterministic fallback when Ollama is offline or unseeded
    return _heuristic_problem_statement(product_context, prospect, research_facts, signals)


def _heuristic_problem_statement(
    product_context: ProductContext,
    prospect: Prospect,
    research_facts: List[str],
    signals: List[Dict[str, Any]],
) -> ProblemStatement:
    """Deterministic fallback statement when local model is offline."""
    primary_pain = product_context.pain_points[0] if product_context.pain_points else f"scaling {product_context.product_name} operations"
    challenge = f"{prospect.company} may be balancing {primary_pain} with existing operational bandwidth."
    
    evidence = []
    if signals:
        evidence.append(f"Signal: {signals[0].get('title', 'hiring activity')}")
    if research_facts:
        evidence.append(research_facts[0][:120])
    if not evidence:
        evidence.append(f"Role alignment: {prospect.title} at {prospect.company}")

    relevance = f"{product_context.product_name} provides {product_context.key_benefits[0] if product_context.key_benefits else 'workflow automation'}."
    uncertainties = ["Current tooling and internal tech priorities", "Immediate quarterly budget allocation"]
    summary = f"Helping {prospect.title}s at {prospect.company} address {primary_pain}."

    return ProblemStatement(
        business_challenge=challenge,
        supporting_evidence=evidence,
        product_relevance=relevance,
        uncertainties=uncertainties,
        summary=summary,
        generation_source="deterministic_heuristic",
    )
