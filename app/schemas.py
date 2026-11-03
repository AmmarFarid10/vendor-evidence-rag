from decimal import Decimal
from typing import Literal

from pydantic import Field

from app.core.http import StrictModel

Attribute = Literal["delivery_days", "warranty_months", "unit_price_usd", "capacity_units"]


class Requirement(StrictModel):
    attribute: Attribute
    operator: Literal["le", "ge"]
    value: Decimal = Field(ge=0, le=100000000)


class Comparison(StrictModel):
    suppliers: list[str] = Field(min_length=1, max_length=8)
    requirements: list[Requirement] = Field(min_length=1, max_length=8)
    extractor: Literal["rules", "ollama"] = "rules"


class Query(StrictModel):
    supplier: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    generate: bool = False
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=1200)
    quote: str = Field(min_length=1, max_length=2000)
    chunk_id: str


class Claims(StrictModel):
    claims: list[Claim] = Field(max_length=8)
