"""Explainable Composite Prospect Scoring for Harvey Local AI Sales Intelligence.

Requirements:
- Evaluates prospects against product-specific ICP (industry, size, geography, job titles, buyer personas, pain points, signals).
- Explainable score with transparent positive factors and deductions.
- Combines proven deterministic heuristics with local semantic similarity.
- Never awards points for missing or unverified data.
- Standardized 0-100 score consistent across products.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sellbird.collectors.people import title_matches
from sellbird.embeddings import prospect_icp_semantic_match
from sellbird.models.prospect import Prospect
from sellbird.product_context import ProductContext

logger = logging.getLogger("sellbird.scoring")

# Heuristic base weights
WEIGHT_SECTOR = 20
WEIGHT_TITLE = 30
WEIGHT_EMAIL = 25
WEIGHT_BUYING_SIGNAL = 25
MAX_SEMANTIC_BOOST = 15

EMAIL_SCORES = {
    "verified": 25,
    "risky": 12,
    "guess": 5,
    "invalid": 0,
}


@dataclass
class ScoringResult:
    """Detailed explainable score breakdown."""
    total_score: int
    score_breakdown: Dict[str, int] = field(default_factory=dict)
    positive_factors: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)
    confidence_level: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "score_breakdown": self.score_breakdown,
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
            "confidence_level": self.confidence_level,
        }

    def explanation_summary(self) -> str:
        pos = "; ".join(self.positive_factors[:3])
        neg = f" (Deductions: {'; '.join(self.negative_factors[:2])})" if self.negative_factors else ""
        return f"Score {self.total_score}/100: {pos}{neg}"


def score_prospect(
    prospect: Prospect,
    product_context: ProductContext,
    company_data: Optional[Dict[str, Any]] = None,
    observations: Optional[List[Dict[str, Any]]] = None,
    semantic_similarity: Optional[float] = None,
) -> ScoringResult:
    """Evaluate a prospect against product context producing an explainable 0-100 score."""
    company = company_data or {}
    obs = observations or []
    breakdown = {}
    positive = []
    negative = []
    score = 0

    # 1. Sector / Industry Match (up to 20 pts)
    target_industries = [i.strip().lower() for i in (product_context.target_industries or []) if i.strip()]
    candidate_ind = (prospect.industry or company.get("industry") or "").strip().lower()
    if candidate_ind and target_industries:
        if any(ind in candidate_ind or candidate_ind in ind for ind in target_industries):
            score += WEIGHT_SECTOR
            breakdown["sector"] = WEIGHT_SECTOR
            positive.append(f"Sector match: '{candidate_ind}' (+{WEIGHT_SECTOR})")
        else:
            negative.append(f"Industry '{candidate_ind}' does not match target ICP industries")
    elif not candidate_ind:
        negative.append("No industry information available (0 points awarded)")

    # 2. Decision Maker Title Match (up to 30 pts)
    target_titles = product_context.job_titles or []
    if prospect.title and target_titles:
        if title_matches(prospect.title, target_titles):
            score += WEIGHT_TITLE
            breakdown["decision_maker"] = WEIGHT_TITLE
            positive.append(f"Target decision-maker role: '{prospect.title}' (+{WEIGHT_TITLE})")
        elif prospect.seniority in ("c_suite", "vp", "director"):
            partial = 15
            score += partial
            breakdown["decision_maker"] = partial
            positive.append(f"Senior executive title (non-primary ICP): '{prospect.title}' (+{partial})")
        else:
            negative.append(f"Title '{prospect.title}' is not an ICP target decision maker")
    elif not prospect.title:
        negative.append("No job title provided (0 points awarded)")

    # 3. Deliverability / Email Verification (up to 25 pts)
    if not (prospect.email and prospect.email.strip()):
        email_pts = 0
        negative.append("No email address provided (0 deliverability points awarded)")
    else:
        status = (prospect.email_status or "guess").lower()
        email_pts = EMAIL_SCORES.get(status, 0)
        score += email_pts
        breakdown["deliverability"] = email_pts
        if email_pts == 25:
            positive.append("Verified email deliverability (+25)")
        elif email_pts == 12:
            positive.append("Accept-all/risky email (+12)")
        elif email_pts == 5:
            positive.append("Pattern-guessed email (+5)")
            negative.append("Email unverified, pattern guess only (-20 potential)")
        else:
            negative.append(f"Invalid email address '{prospect.email}' (0 deliverability points)")

    # 4. Verified Buying Signals (up to 25 pts)
    hiring_terms = [s.strip().lower() for s in product_context.buying_signals if s.strip()]
    matched_signal = None
    for o in obs:
        text = (o.get("value_text") or "").lower()
        if o.get("signal_code") in ("HIRING_ROLE", "BUYING_SIGNAL", "TECH_STACK"):
            for term in hiring_terms:
                if term in text:
                    matched_signal = o.get("value_text")
                    break
        if matched_signal:
            break

    if matched_signal:
        score += WEIGHT_BUYING_SIGNAL
        breakdown["buying_signal"] = WEIGHT_BUYING_SIGNAL
        positive.append(f"Verified buying signal: '{matched_signal[:80]}' (+{WEIGHT_BUYING_SIGNAL})")
    elif any(o.get("signal_code") == "HIRING_ROLE" for o in obs):
        partial_sig = 5
        score += partial_sig
        breakdown["buying_signal"] = partial_sig
        positive.append(f"Active hiring detected, no exact ICP role match (+{partial_sig})")
    else:
        negative.append("No active buying signals observed (0 points awarded)")

    # 5. Semantic ICP Alignment Boost (up to 15 pts)
    # Anti-inflation rule: Semantic similarity must never unfairly inflate weak prospects.
    # It requires at least one baseline qualification match (sector, role, or verified buying signal).
    has_baseline_qualification = (
        breakdown.get("sector", 0) > 0
        or breakdown.get("decision_maker", 0) > 0
        or (breakdown.get("buying_signal", 0) > 0 and matched_signal is not None)
    )

    sim = semantic_similarity
    if sim is None and (prospect.title or candidate_ind):
        profile_snippet = f"{prospect.title} at {prospect.company or 'a company'} in {candidate_ind}."
        sim, _ = prospect_icp_semantic_match(
            prospect_profile=profile_snippet,
            icp_description=product_context.product_description,
            target_roles=product_context.job_titles,
            pain_points=product_context.pain_points,
        )

    if sim is not None and sim >= 0.65:
        if has_baseline_qualification:
            raw_boost = int(round(sim * MAX_SEMANTIC_BOOST))
            # Sensible capping: Total score and breakdown sum must never exceed 100
            applied_boost = max(0, min(raw_boost, 100 - score))
            score += applied_boost
            if applied_boost > 0:
                breakdown["semantic_similarity"] = applied_boost
                if applied_boost < raw_boost:
                    positive.append(
                        f"Semantic alignment boost: {sim:.2f} cosine similarity "
                        f"(+{applied_boost}, capped to 100 max total score)"
                    )
                else:
                    positive.append(f"Semantic alignment boost: {sim:.2f} cosine similarity (+{applied_boost})")
            else:
                positive.append(f"Semantic alignment verified: {sim:.2f} cosine similarity (score already at 100 cap)")
        else:
            negative.append(
                f"Semantic similarity ({sim:.2f}) not applied: prospect lacks baseline sector, role, or signal qualification"
            )

    # Final bounded score: strictly bounded between 0 and 100
    final_score = max(0, min(100, score))

    # Confidence calculation
    if email_pts == 25 and matched_signal:
        confidence = "high"
    elif email_pts >= 12:
        confidence = "medium"
    else:
        confidence = "low"

    return ScoringResult(
        total_score=final_score,
        score_breakdown=breakdown,
        positive_factors=positive,
        negative_factors=negative,
        confidence_level=confidence,
    )
