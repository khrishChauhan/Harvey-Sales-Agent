"""Business profile data model."""

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Naive UTC now (consistent with DB storage)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class BusinessProfile(BaseModel):
    id: str = ""
    name: str
    description: str = ""
    website: str = ""
    industry: str = ""
    company_size: str = ""
    business_location: str = ""
    target_markets: list[str] = Field(default_factory=list)
    brand_voice: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def target_markets_json(self) -> str:
        return json.dumps(self.target_markets)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "BusinessProfile":
        d = dict(row)
        raw_markets = d.get("target_markets")
        if isinstance(raw_markets, str):
            try:
                d["target_markets"] = json.loads(raw_markets) if raw_markets else []
            except (json.JSONDecodeError, TypeError):
                d["target_markets"] = [raw_markets] if raw_markets else []
        elif not isinstance(raw_markets, list):
            d["target_markets"] = []
        return cls(**d)
