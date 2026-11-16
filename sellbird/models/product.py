"""Product data model."""

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Naive UTC now (consistent with DB storage)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Product(BaseModel):
    id: str = ""
    business_id: str = ""
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
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def key_features_json(self) -> str:
        return json.dumps(self.key_features)

    def key_benefits_json(self) -> str:
        return json.dumps(self.key_benefits)

    def unique_selling_points_json(self) -> str:
        return json.dumps(self.unique_selling_points)

    def target_industries_json(self) -> str:
        return json.dumps(self.target_industries)

    def objection_responses_json(self) -> str:
        return json.dumps(self.objection_responses)

    def offer_json(self) -> str:
        return json.dumps(self.offer)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Product":
        d = dict(row)
        for list_col in ("key_features", "key_benefits", "unique_selling_points", "target_industries"):
            val = d.get(list_col)
            if isinstance(val, str):
                try:
                    d[list_col] = json.loads(val) if val else []
                except (json.JSONDecodeError, TypeError):
                    d[list_col] = [val] if val else []
            elif not isinstance(val, list):
                d[list_col] = []

        for dict_col in ("objection_responses", "offer"):
            val = d.get(dict_col)
            if isinstance(val, str):
                try:
                    d[dict_col] = json.loads(val) if val else {}
                except (json.JSONDecodeError, TypeError):
                    d[dict_col] = {}
            elif not isinstance(val, dict):
                d[dict_col] = {}

        return cls(**d)
