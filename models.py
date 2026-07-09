"""
models.py
---------
Pydantic models for structured intent parsing and validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from config import OPERATIONS

class ConditionModel(BaseModel):
    column: str | None = None
    op: str | None = None
    value: Any | None = None
    value2: Any | None = None

    @field_validator("op")
    @classmethod
    def check_op(cls, v: str | None) -> str | None:
        if v:
            val = v.lower().strip()
            allowed = {"eq", "neq", "gt", "gte", "lt", "lte", "between", "contains", "like", "isna", "notna", "==", ">", ">=", "<", "<="}
            if val not in allowed:
                raise ValueError(f"Invalid condition op '{val}'. Allowed: {sorted(allowed)}")
            return val
        return v

class SingleIntentModel(BaseModel):
    operation: str | None = None
    column: str | None = None
    group_by: str | None = None
    agg_column: str | None = None
    agg_func: str | None = None
    value: Any | None = None
    value2: Any | None = None
    n: int | None = None
    ascending: bool | None = None
    date_op: str | None = None
    conditions: list[ConditionModel] = Field(default_factory=list)

    @field_validator("operation")
    @classmethod
    def check_operation(cls, v: str | None) -> str | None:
        if v:
            val = v.lower().strip()
            if val not in OPERATIONS:
                raise ValueError(f"Invalid operation '{val}'. Allowed: {sorted(OPERATIONS)}")
            return val
        return v

class RootIntentModel(SingleIntentModel):
    """
    The top-level intent.
    It can either be a single operation (using fields from SingleIntentModel)
    or a multi-step query containing a list of SingleIntentModel steps.
    """
    steps: list[SingleIntentModel] | None = None
