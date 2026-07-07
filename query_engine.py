"""
query_engine.py
---------------
Deterministic execution engine. This module NEVER asks an LLM for a
number. It takes a structured "intent" dictionary (either produced by
Gemini's intent-understanding call, or by the local rule-based
fallback parser) and executes it with pandas only.

Every result carries an `explanation` string describing exactly which
pandas operation produced it, so the UI can show "How was this answer
calculated?" with full transparency.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config import OPERATIONS
from utils import DatasetSchema, resolve_column

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Encapsulates the outcome of a deterministic pandas query."""
    operation: str
    success: bool
    scalar_result: Any = None
    table_result: pd.DataFrame | None = None
    explanation: str = ""
    error: str | None = None
    columns_used: list[str] = field(default_factory=list)
    confidence: float = 1.0       # 1.0 = fully deterministic pandas result
    execution_time_ms: float = 0  # wall-clock time for the pandas operation
    filters_applied: int = 0      # number of filter conditions applied
    rows_scanned: int = 0         # rows in the source before filters
    rows_matched: int = 0         # rows after filters


class QueryEngine:
    """Executes structured intents against a single dataframe + schema."""

    def __init__(self, df: pd.DataFrame, schema: DatasetSchema) -> None:
        self.df = df
        self.schema = schema

    # ------------------------------------------------------------------
    # Column resolution helpers
    # ------------------------------------------------------------------
    def _resolve(self, col_hint: str | None, default: str | None = None) -> str | None:
        if not col_hint:
            return default
        if col_hint in self.df.columns:
            return col_hint
        resolved = resolve_column(col_hint, list(self.df.columns))
        return resolved or default

    def _numeric_col(self, col_hint: str | None) -> str | None:
        col = self._resolve(col_hint, self.schema.primary_budget_col)
        if col and col in self.df.select_dtypes(include="number").columns:
            return col
        # try any numeric column as last resort
        numeric_cols = list(self.df.select_dtypes(include="number").columns)
        return numeric_cols[0] if numeric_cols else None

    def _apply_conditions(
        self, df: pd.DataFrame, conditions: list[dict]
    ) -> tuple[pd.DataFrame, list[str]]:
        used: list[str] = []
        out = df
        for cond in conditions or []:
            col = self._resolve(cond.get("column"))
            op = (cond.get("op") or "eq").lower()
            val = cond.get("value")
            val2 = cond.get("value2")
            if not col:
                continue
            used.append(col)
            series = out[col]
            try:
                if op in {"eq", "=="}:
                    if pd.api.types.is_numeric_dtype(series):
                        out = out[series == float(val)]
                    else:
                        out = out[series.astype(str).str.lower() == str(val).lower()]
                elif op in {"contains", "like"}:
                    out = out[series.astype(str).str.lower().str.contains(str(val).lower(), regex=False, na=False)]
                elif op in {"gt", ">"}:
                    out = out[pd.to_numeric(series, errors="coerce") > float(val)]
                elif op in {"gte", ">="}:
                    out = out[pd.to_numeric(series, errors="coerce") >= float(val)]
                elif op in {"lt", "<"}:
                    out = out[pd.to_numeric(series, errors="coerce") < float(val)]
                elif op in {"lte", "<="}:
                    out = out[pd.to_numeric(series, errors="coerce") <= float(val)]
                elif op == "between":
                    lo, hi = sorted([float(val), float(val2)])
                    numeric = pd.to_numeric(series, errors="coerce")
                    out = out[(numeric >= lo) & (numeric <= hi)]
                elif op == "neq":
                    out = out[series.astype(str).str.lower() != str(val).lower()]
            except (ValueError, TypeError):
                continue
        return out, used

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def execute(self, intent: dict) -> QueryResult:
        """Execute a structured intent and return a QueryResult."""
        op = (intent.get("operation") or "").lower().strip()
        if op not in OPERATIONS:
            return QueryResult(
                operation=op or "unknown",
                success=False,
                error=(
                    f"Unrecognized operation '{op}'. Try asking about counts, "
                    f"averages, filters, sorting, or grouping."
                ),
            )
        try:
            handler = getattr(self, f"_op_{op}")
        except AttributeError:
            return QueryResult(
                operation=op, success=False,
                error=f"No handler implemented for '{op}'.",
            )

        t0 = time.time()
        try:
            result = handler(intent)
            result.execution_time_ms = round((time.time() - t0) * 1000, 1)
            result.rows_scanned = len(self.df)
            result.filters_applied = len(intent.get("conditions", []))
            if result.table_result is not None:
                result.rows_matched = len(result.table_result)
            elif result.scalar_result is not None:
                result.rows_matched = result.scalar_result if isinstance(result.scalar_result, int) else 0
            logger.info(
                "Query executed: op=%s, time=%.1fms, rows_matched=%d",
                op, result.execution_time_ms, result.rows_matched,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = round((time.time() - t0) * 1000, 1)
            logger.error("Query failed: op=%s, error=%s, time=%.1fms", op, exc, elapsed)
            return QueryResult(
                operation=op, success=False,
                error=f"Could not execute query: {exc}",
                execution_time_ms=elapsed,
            )

    # ------------------------------------------------------------------
    # Operation handlers – each returns a QueryResult
    # ------------------------------------------------------------------
    def _op_count(self, intent: dict) -> QueryResult:
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))
        n = len(filtered)
        return QueryResult(
            operation="count", success=True, scalar_result=n,
            table_result=filtered, columns_used=used,
            explanation=f"Counted rows in the dataframe after applying "
                        f"{len(intent.get('conditions', []))} filter(s): {n} rows matched.",
        )

    def _op_sum(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.get("column"))
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))
        total = pd.to_numeric(filtered[col], errors="coerce").sum() if col else None
        return QueryResult(
            operation="sum", success=True, scalar_result=total, table_result=filtered,
            columns_used=[col] + used if col else used,
            explanation=f"Computed df['{col}'].sum() after filters -> {total}.",
        )

    def _op_average(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.get("column"))
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))
        avg = pd.to_numeric(filtered[col], errors="coerce").mean() if col else None
        return QueryResult(
            operation="average", success=True, scalar_result=avg, table_result=filtered,
            columns_used=[col] + used if col else used,
            explanation=f"Computed df['{col}'].mean() after filters -> {avg}.",
        )

    def _op_extremum(self, intent: dict, func: str) -> QueryResult:
        """Shared handler for min/max operations to avoid duplication."""
        col = self._numeric_col(intent.get("column"))
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))
        series = pd.to_numeric(filtered[col], errors="coerce") if col else None
        if series is not None and not series.empty:
            val = getattr(series, func)()
            idx_func = f"idx{func}"
            row = filtered.loc[[getattr(series, idx_func)()]]
        else:
            val = None
            row = filtered.head(0)
        return QueryResult(
            operation=func, success=True, scalar_result=val, table_result=row,
            columns_used=[col] + used if col else used,
            explanation=f"Computed df['{col}'].{func}() after filters -> {val}.",
        )

    def _op_min(self, intent: dict) -> QueryResult:
        return self._op_extremum(intent, "min")

    def _op_max(self, intent: dict) -> QueryResult:
        return self._op_extremum(intent, "max")

    def _op_filter(self, intent: dict) -> QueryResult:
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))
        return QueryResult(
            operation="filter", success=True, table_result=filtered, columns_used=used,
            scalar_result=len(filtered),
            explanation=f"Applied {len(intent.get('conditions', []))} filter condition(s) "
                        f"-> {len(filtered)} matching rows.",
        )

    def _op_greater_than(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.get("column"))
        val = intent.get("value")
        conditions = [{"column": col, "op": "gt", "value": val}] + list(intent.get("conditions", []))
        filtered, used = self._apply_conditions(self.df, conditions)
        return QueryResult(
            operation="greater_than", success=True, table_result=filtered,
            scalar_result=len(filtered), columns_used=[col] + used,
            explanation=f"Filtered rows where {col} > {val} -> {len(filtered)} rows.",
        )

    def _op_less_than(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.get("column"))
        val = intent.get("value")
        conditions = [{"column": col, "op": "lt", "value": val}] + list(intent.get("conditions", []))
        filtered, used = self._apply_conditions(self.df, conditions)
        return QueryResult(
            operation="less_than", success=True, table_result=filtered,
            scalar_result=len(filtered), columns_used=[col] + used,
            explanation=f"Filtered rows where {col} < {val} -> {len(filtered)} rows.",
        )

    def _op_between(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.get("column"))
        val, val2 = intent.get("value"), intent.get("value2")
        conditions = [{"column": col, "op": "between", "value": val, "value2": val2}] + list(intent.get("conditions", []))
        filtered, used = self._apply_conditions(self.df, conditions)
        return QueryResult(
            operation="between", success=True, table_result=filtered,
            scalar_result=len(filtered), columns_used=[col] + used,
            explanation=f"Filtered rows where {col} is between {val} and {val2} "
                        f"-> {len(filtered)} rows.",
        )

    def _op_sort(self, intent: dict) -> QueryResult:
        col = self._resolve(intent.get("column"), self.schema.primary_budget_col)
        ascending = bool(intent.get("ascending", False))
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))
        sorted_df = filtered.sort_values(by=col, ascending=ascending) if col else filtered
        n = intent.get("n")
        if n:
            sorted_df = sorted_df.head(int(n))
        return QueryResult(
            operation="sort", success=True, table_result=sorted_df,
            columns_used=[col] + used,
            explanation=f"Sorted rows by '{col}' ({'ascending' if ascending else 'descending'})."
                        + (f" Limited to top {n}." if n else ""),
        )

    def _op_ranked(self, intent: dict, ascending: bool) -> QueryResult:
        """Shared handler for topn/bottomn to avoid duplication."""
        col = self._numeric_col(intent.get("column"))
        n = int(intent.get("n") or 5)
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))
        ranked = filtered.sort_values(by=col, ascending=ascending).head(n) if col else filtered.head(n)
        label = "bottom" if ascending else "top"
        return QueryResult(
            operation=f"{label}n", success=True, table_result=ranked,
            columns_used=[col] + used,
            explanation=f"Sorted by '{col}' {'ascending' if ascending else 'descending'} "
                        f"and took the {label} {n} rows.",
        )

    def _op_topn(self, intent: dict) -> QueryResult:
        return self._op_ranked(intent, ascending=False)

    def _op_bottomn(self, intent: dict) -> QueryResult:
        return self._op_ranked(intent, ascending=True)

    def _op_groupby(self, intent: dict) -> QueryResult:
        group_col = self._resolve(
            intent.get("group_by") or intent.get("column"),
            self.schema.location_col or self.schema.property_type_col,
        )
        agg_col = self._numeric_col(intent.get("agg_column"))
        agg_func = (intent.get("agg_func") or "mean").lower()
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))

        if not group_col:
            return QueryResult(
                operation="groupby", success=False,
                error="Could not determine a column to group by.",
            )

        if agg_func not in {"mean", "sum", "count", "min", "max", "median"}:
            agg_func = "mean"

        if agg_func == "count" or agg_col is None:
            grouped = filtered.groupby(group_col).size().reset_index(name="count")
            grouped = grouped.sort_values(by="count", ascending=False)
            explanation = f"Grouped by '{group_col}' and counted rows in each group."
        else:
            grouped = (
                filtered.groupby(group_col)[agg_col]
                .agg(agg_func)
                .reset_index()
                .sort_values(by=agg_col, ascending=False)
            )
            explanation = f"Grouped by '{group_col}' and computed {agg_func}() of '{agg_col}' per group."

        return QueryResult(
            operation="groupby", success=True, table_result=grouped,
            columns_used=[group_col, agg_col] if agg_col else [group_col],
            explanation=explanation,
        )

    def _op_unique(self, intent: dict) -> QueryResult:
        col = self._resolve(intent.get("column"), self.schema.location_col)
        values = sorted(self.df[col].dropna().astype(str).unique().tolist()) if col else []
        return QueryResult(
            operation="unique", success=True,
            table_result=pd.DataFrame({col: values}) if col else None,
            scalar_result=values, columns_used=[col] if col else [],
            explanation=f"Computed df['{col}'].unique() -> {len(values)} distinct values.",
        )

    def _op_distinct_count(self, intent: dict) -> QueryResult:
        col = self._resolve(intent.get("column"), self.schema.location_col)
        n = int(self.df[col].nunique(dropna=True)) if col else None
        return QueryResult(
            operation="distinct_count", success=True, scalar_result=n,
            columns_used=[col] if col else [],
            explanation=f"Computed df['{col}'].nunique() -> {n}.",
        )

    def _op_describe(self, intent: dict) -> QueryResult:
        col = self._resolve(intent.get("column"))
        target = self.df[[col]] if col else self.df.select_dtypes(include="number")
        desc = target.describe()
        return QueryResult(
            operation="describe", success=True, table_result=desc,
            columns_used=[col] if col else [],
            explanation="Computed df.describe() summary statistics.",
        )

    def _op_list(self, intent: dict) -> QueryResult:
        filtered, used = self._apply_conditions(self.df, intent.get("conditions", []))
        n = intent.get("n")
        if n:
            filtered = filtered.head(int(n))
        return QueryResult(
            operation="list", success=True, table_result=filtered, columns_used=used,
            scalar_result=len(filtered),
            explanation=f"Listed {len(filtered)} matching row(s) after applying filters.",
        )


# --------------------------------------------------------------------------
# Rule-based fallback intent parser (used if Gemini is unavailable / fails)
# --------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(lakh|lac|crore|cr|k|thousand)?", re.IGNORECASE)

_MULTIPLIERS = {
    "lakh": 100_000, "lac": 100_000, "crore": 10_000_000, "cr": 10_000_000,
    "k": 1_000, "thousand": 1_000,
}


def _parse_number_with_unit(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "").lower()
    return value * _MULTIPLIERS.get(unit, 1)


def _extract_numbers_with_shared_units(q: str) -> list[float]:
    """Parse numbers like '80 and 120 lakhs' where the unit trails the last
    number but should apply to all of them (a common natural-language
    pattern for ranges)."""
    raw_matches = list(_NUMBER_RE.finditer(q))
    if not raw_matches:
        return []
    trailing_unit = None
    for m in reversed(raw_matches):
        if m.group(2):
            trailing_unit = m.group(2).lower()
            break
    results = []
    for m in raw_matches:
        value = float(m.group(1))
        unit = (m.group(2) or trailing_unit or "").lower()
        results.append(value * _MULTIPLIERS.get(unit, 1))
    return results


def _match_categorical_conditions(
    q: str, schema: DatasetSchema, df: pd.DataFrame | None = None
) -> list[dict]:
    """Detect mentions of known categorical values (e.g. '2BHK', 'Pune')
    anywhere in the question and turn them into filter conditions, without
    hardcoding any specific column name."""
    conditions: list[dict] = []
    if df is None:
        return conditions
    candidate_cols = [
        c for c in [schema.property_type_col, schema.location_col, schema.status_col] if c
    ]
    for col in candidate_cols:
        if col not in df.columns:
            continue
        for value in df[col].dropna().astype(str).unique():
            token = value.lower().strip()
            if token and re.search(rf"\b{re.escape(token)}\b", q):
                conditions.append({"column": col, "op": "contains", "value": value})
                break  # only take the first matching value per column
    return conditions


def merge_follow_up_conditions(
    new_intent: dict, previous_conditions: list[dict]
) -> dict:
    """Merge conditions from a previous query into the new intent to
    support conversational follow-ups like 'only those above 90 lakhs'.

    Only merges if the new intent has no conditions of its own for a
    column that the previous query already filtered on.
    """
    if not previous_conditions:
        return new_intent

    existing = new_intent.get("conditions", [])
    existing_cols = {c.get("column") for c in existing}
    for cond in previous_conditions:
        if cond.get("column") not in existing_cols:
            existing.append(cond)
    new_intent["conditions"] = existing
    return new_intent


def rule_based_intent(
    question: str, schema: DatasetSchema, df: pd.DataFrame | None = None
) -> dict[str, Any]:
    """A conservative keyword-based fallback so the app still works
    without an LLM (or if the Gemini call fails / quota is exceeded).
    """
    q = question.lower()
    budget_col = schema.primary_budget_col
    categorical_conditions = _match_categorical_conditions(q, schema, df)
    numbers = _extract_numbers_with_shared_units(q)

    if "unique" in q or "distinct" in q:
        col = schema.location_col if ("location" in q or "city" in q or "area" in q) else (
            schema.property_type_col if "type" in q else schema.location_col
        )
        if "how many" in q or "count" in q:
            return {"operation": "distinct_count", "column": col}
        return {"operation": "unique", "column": col}
    if "average" in q or "avg" in q or "mean" in q:
        if ("by" in q or "each" in q or "per " in q) and (schema.location_col or schema.property_type_col):
            group_col = schema.location_col if ("city" in q or "location" in q) else (schema.property_type_col or schema.location_col)
            return {"operation": "groupby", "group_by": group_col, "agg_column": budget_col, "agg_func": "mean"}
        return {"operation": "average", "column": budget_col, "conditions": categorical_conditions}
    if "sum" in q or "total" in q:
        return {"operation": "sum", "column": budget_col, "conditions": categorical_conditions}
    if "highest" in q or "max" in q or "maximum" in q:
        if ("by" in q or "which" in q) and (schema.location_col or schema.property_type_col):
            group_col = schema.location_col or schema.property_type_col
            return {"operation": "groupby", "group_by": group_col, "agg_column": budget_col, "agg_func": "mean"}
        return {"operation": "max", "column": budget_col, "conditions": categorical_conditions}
    if "lowest" in q or "min" in q or "minimum" in q or "cheapest" in q:
        return {"operation": "min", "column": budget_col, "conditions": categorical_conditions}
    if "between" in q and len(numbers) >= 2:
        return {"operation": "between", "column": budget_col, "value": numbers[0], "value2": numbers[1], "conditions": categorical_conditions}
    if ("above" in q or "greater" in q or "more than" in q or "over" in q) and numbers:
        return {"operation": "greater_than", "column": budget_col, "value": numbers[0], "conditions": categorical_conditions}
    if ("below" in q or "less than" in q or "under" in q) and numbers:
        return {"operation": "less_than", "column": budget_col, "value": numbers[0], "conditions": categorical_conditions}
    if "top" in q:
        n_match = re.search(r"top\s+(\d+)", q)
        return {"operation": "topn", "column": budget_col, "n": int(n_match.group(1)) if n_match else 5, "conditions": categorical_conditions}
    if "bottom" in q:
        n_match = re.search(r"bottom\s+(\d+)", q)
        return {"operation": "bottomn", "column": budget_col, "n": int(n_match.group(1)) if n_match else 5, "conditions": categorical_conditions}
    if "group" in q or "breakdown" in q:
        group_col = schema.status_col or schema.location_col or schema.property_type_col
        return {"operation": "groupby", "group_by": group_col, "agg_column": None, "agg_func": "count"}
    if "how many" in q or "count" in q or "number of" in q:
        return {"operation": "count", "conditions": categorical_conditions}
    if categorical_conditions or "list" in q or "show" in q or "give me" in q or "interested" in q:
        return {"operation": "list", "conditions": categorical_conditions, "n": 50}
    return {"operation": "list", "conditions": categorical_conditions, "n": 20}
