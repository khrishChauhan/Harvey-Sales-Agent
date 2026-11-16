"""Ideal Customer Profile (ICP) data model."""

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Naive UTC now (consistent with DB storage)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ICP(BaseModel):
    id: str = ""
    product_id: str = ""
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
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ICP":
        d = dict(row)
        for col in (
            "target_industries", "company_sizes", "target_geographies",
            "job_titles", "buyer_personas", "pain_points",
            "buying_signals", "qualification_criteria",
        ):
            val = d.get(col)
            if isinstance(val, str):
                try:
                    d[col] = json.loads(val) if val else []
                except (json.JSONDecodeError, TypeError):
                    d[col] = [val] if val else []
            elif not isinstance(val, list):
                d[col] = []
        return cls(**d)
