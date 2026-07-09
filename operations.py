"""
operations.py
-------------
Deterministic execution engine. This module NEVER asks an LLM for a
number. It takes a structured "intent" dictionary (either produced by
Gemini's intent-understanding call, or by the local rule-based
fallback parser) and executes it with pandas only.

Every result carries an `explanation` string describing exactly which
pandas operation produced it, so the UI can show "How was this answer
calculated?" with full transparency.
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from config import (
    MAX_QUERY_STEPS,
    OPERATIONS,
)
from matching import fuzzy_match_category
from models import RootIntentModel, SingleIntentModel, ConditionModel
from utils import DatasetSchema, resolve_column

logger = logging.getLogger(__name__)

__all__ = [
    "QueryEngine",
    "QueryResult",
    "StepTrace",
    "merge_follow_up_conditions",
    "rule_based_intent",
    "validate_chain_intent",
]

@dataclass
class StepTrace:
    """Summary of a single step in a chained query execution."""
    step_number: int
    operation: str
    rows_before: int
    rows_after: int
    explanation: str
    execution_time_ms: float


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
    step_trace: list[StepTrace] = field(default_factory=list)  # populated by execute_chain()
    execution_path: str = "fixed"          # "fixed" or "dynamic"
    dynamic_code: str | None = None        # generated code (if dynamic path)
    dynamic_retry_count: int = 0           # number of retries used
    fuzzy_matches: list[dict] = field(default_factory=list)  # list of dicts: original, matched, score, method


class QueryEngine:
    """Executes structured intents against a single dataframe + schema."""

    def __init__(self, df: pd.DataFrame, schema: DatasetSchema, value_index: dict | None = None) -> None:
        self.df = df
        self.schema = schema
        self.value_index = value_index or {}
        self.fuzzy_matches_recorded: list[dict] = []

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
        self, df: pd.DataFrame, conditions: list[Any]
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Apply filter conditions and return (filtered_df, used_cols, skipped).

        The *skipped* list records any conditions that could not be applied
        (e.g. type conversion failure) so the UI can surface them.
        """
        used: list[str] = []
        skipped: list[str] = []
        out = df
        for cond in conditions or []:
            if isinstance(cond, dict):
                cond = ConditionModel.model_validate(cond)
            col = self._resolve(cond.column)
            op = (cond.op or "eq").lower()
            val = cond.value
            val2 = cond.value2
            if not col:
                skipped.append(f"Unresolved column '{cond.column}'")
                continue
            used.append(col)
            series = out[col]
            try:
                # --------------------------------------------------------------
                # Phase 3: Fuzzy Categorical Matching
                # Intercept exact matches on categorical columns
                # --------------------------------------------------------------
                fuzzy_matched = False
                if op in {"eq", "==", "neq"} and self.value_index and col in self.schema.categorical_cols:
                    match_res = fuzzy_match_category(val, col, self.value_index)
                    if match_res:
                        val = match_res["matched"]
                        fuzzy_matched = True
                        if match_res["method"] != "exact":
                            self.fuzzy_matches_recorded.append(match_res)

                def _safe_float(v: Any) -> float:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        if isinstance(v, str):
                            parsed = _parse_number_with_unit(v)
                            if parsed is not None:
                                return parsed
                        raise

                if op in {"eq", "=="}:
                    if pd.api.types.is_numeric_dtype(series):
                        out = out[series == _safe_float(val)]
                    else:
                        if fuzzy_matched:
                            out = out[series.astype(str) == str(val)]
                        else:
                            out = out[series.astype(str).str.lower() == str(val).lower()]
                elif op in {"contains", "like"}:
                    out = out[series.astype(str).str.lower().str.contains(str(val).lower(), regex=False, na=False)]
                elif op in {"gt", ">"}:
                    out = out[pd.to_numeric(series, errors="coerce") > _safe_float(val)]
                elif op in {"gte", ">="}:
                    out = out[pd.to_numeric(series, errors="coerce") >= _safe_float(val)]
                elif op in {"lt", "<"}:
                    out = out[pd.to_numeric(series, errors="coerce") < _safe_float(val)]
                elif op in {"lte", "<="}:
                    out = out[pd.to_numeric(series, errors="coerce") <= _safe_float(val)]
                elif op == "between":
                    lo, hi = sorted([_safe_float(val), _safe_float(val2)])
                    numeric = pd.to_numeric(series, errors="coerce")
                    out = out[(numeric >= lo) & (numeric <= hi)]
                elif op == "neq":
                    if fuzzy_matched:
                        out = out[series.astype(str) != str(val)]
                    else:
                        out = out[series.astype(str).str.lower() != str(val).lower()]
                elif op == "isna":
                    out = out[series.isna()]
                elif op == "notna":
                    out = out[series.notna()]
                else:
                    skipped.append(f"Unknown operator '{op}' for column '{col}'")
            except (ValueError, TypeError) as exc:
                skipped.append(f"Filter on '{col}' {op} '{val}' failed: {exc}")
                continue
        return out, used, skipped

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def execute(self, intent: Any) -> QueryResult:
        """Execute a structured intent and return a QueryResult."""
        try:
            if isinstance(intent, dict):
                intent = RootIntentModel.model_validate(intent)
        except ValidationError as e:
            # We catch validation errors (e.g. invalid operations like 'nonexistent')
            # and return a gracefully failed query result so the dynamic fallback can take over.
            return QueryResult(
                operation=intent.get('operation', 'unknown') if isinstance(intent, dict) else 'unknown',
                success=False,
                error=(
                    f"Unrecognized operation or invalid schema: {e}. Try asking about counts, "
                    f"averages, filters, sorting, or grouping."
                ),
            )
            
        op = (intent.operation or "").lower().strip()
        
        # NORMALIZATION: 
        # If Gemini placed a filter condition in the top-level fields 
        # for an operation that natively ignores them, move it into conditions.
        if op in {"count", "filter", "list", "groupby"}:
            if getattr(intent, "column", None) and getattr(intent, "value", None) is not None:
                # Avoid duplicates
                existing = {(c.column, c.value) for c in getattr(intent, "conditions", [])}
                if (intent.column, intent.value) not in existing:
                    from models import ConditionModel
                    if not hasattr(intent, "conditions"):
                        intent.conditions = []
                    intent.conditions.append(ConditionModel(
                        column=intent.column,
                        op="eq",
                        value=intent.value,
                        value2=getattr(intent, "value2", None)
                    ))
                # Clear them so they aren't misused
                intent.column = None
                intent.value = None
                intent.value2 = None
                
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
            result.filters_applied = len(intent.conditions)
            if result.table_result is not None:
                result.rows_matched = len(result.table_result)
            elif result.scalar_result is not None:
                result.rows_matched = result.scalar_result if isinstance(result.scalar_result, int) else 0
            
            # Attach fuzzy matches recorded during this execution
            result.fuzzy_matches.extend(self.fuzzy_matches_recorded)
            self.fuzzy_matches_recorded.clear()

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
    # Chained execution
    # ------------------------------------------------------------------
    def execute_chain(self, steps: list[dict]) -> QueryResult:
        """Execute a sequence of chained operations.

        Each step is a standard intent dict (same shape as accepted by
        ``execute()``).  The table result of step *i* becomes the working
        DataFrame for step *i + 1*.  The final step's ``QueryResult`` is
        returned with a populated ``step_trace`` list so the UI can
        display the full audit trail.

        Supported intent shapes (both are accepted):

        Legacy single-operation (routed through ``execute()`` directly)::

            return SingleIntentModel(
                operation="filter",
                conditions=categorical_conditions
            )    {"operation": "sort", "column": "Budget", "ascending": false},
                {"operation": "topn", "column": "Budget", "n": 5}
            ]}
        """
        trace: list[StepTrace] = []
        working_df = self.df
        final_result: QueryResult | None = None
        t0_chain = time.time()

        for i, step in enumerate(steps):
            if isinstance(step, dict):
                step = SingleIntentModel.model_validate(step)
            rows_before = len(working_df)
            step_engine = QueryEngine(working_df, self.schema)
            result = step_engine.execute(step)

            rows_after = (
                len(result.table_result)
                if result.table_result is not None
                else 0
            )
            trace.append(StepTrace(
                step_number=i + 1,
                operation=(step.operation or "unknown"),
                rows_before=rows_before,
                rows_after=rows_after,
                explanation=result.explanation,
                execution_time_ms=result.execution_time_ms,
            ))

            if not result.success:
                result.step_trace = trace
                result.error = (
                    f"Step {i + 1} ({(step.operation or "unknown")}) failed: "
                    f"{result.error}"
                )
                logger.warning(
                    "Chain aborted at step %d/%d: %s",
                    i + 1, len(steps), result.error,
                )
                return result

            # Propagate table to next step
            if result.table_result is not None:
                working_df = result.table_result

            final_result = result

        # Should never be None here (steps is validated non-empty), but
        # guard defensively.
        if final_result is None:  # pragma: no cover
            return QueryResult(
                operation="chain", success=False,
                error="Chain produced no result (empty steps list).",
            )

        final_result.step_trace = trace
        final_result.execution_time_ms = round(
            (time.time() - t0_chain) * 1000, 1,
        )
        final_result.rows_scanned = len(self.df)
        logger.info(
            "Chain executed: %d steps, total_time=%.1fms",
            len(steps), final_result.execution_time_ms,
        )
        return final_result

    # ------------------------------------------------------------------
    # Operation handlers – each returns a QueryResult
    # ------------------------------------------------------------------
    def _op_count(self, intent: dict) -> QueryResult:
        filtered, used, skipped = self._apply_conditions(self.df, intent.conditions)
        n = len(filtered)
        explanation = (
            f"Counted rows in the dataframe after applying "
            f"{len(intent.conditions)} filter(s): {n} rows matched."
        )
        if skipped:
            explanation += f" Skipped conditions: {'; '.join(skipped)}."
        return QueryResult(
            operation="count", success=True, scalar_result=n,
            table_result=filtered, columns_used=used,
            explanation=explanation,
        )

    def _op_sum(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.column)
        filtered, used, skipped = self._apply_conditions(self.df, intent.conditions)
        series = pd.to_numeric(filtered[col], errors="coerce") if col else None
        nan_excluded = int(series.isna().sum()) if series is not None else 0
        total = series.sum() if series is not None else None
        explanation = f"Computed df['{col}'].sum() after filters -> {total}."
        if nan_excluded:
            explanation += f" ({nan_excluded} NaN values excluded)"
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation="sum", success=True, scalar_result=total, table_result=filtered,
            columns_used=[col] + used if col else used,
            explanation=explanation,
        )

    def _op_average(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.column)
        filtered, used, skipped = self._apply_conditions(self.df, intent.conditions)
        series = pd.to_numeric(filtered[col], errors="coerce") if col else None
        nan_excluded = int(series.isna().sum()) if series is not None else 0
        avg = series.mean() if series is not None else None
        explanation = f"Computed df['{col}'].mean() after filters -> {avg}."
        if nan_excluded:
            explanation += f" ({nan_excluded} NaN values excluded)"
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation="average", success=True, scalar_result=avg, table_result=filtered,
            columns_used=[col] + used if col else used,
            explanation=explanation,
        )

    def _op_median(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.column)
        filtered, used, skipped = self._apply_conditions(self.df, intent.conditions)
        series = pd.to_numeric(filtered[col], errors="coerce") if col else None
        nan_excluded = int(series.isna().sum()) if series is not None else 0
        med = series.median() if series is not None else None
        explanation = f"Computed df['{col}'].median() after filters -> {med}."
        if nan_excluded:
            explanation += f" ({nan_excluded} NaN values excluded)"
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation="median", success=True, scalar_result=med, table_result=filtered,
            columns_used=[col] + used if col else used,
            explanation=explanation,
        )

    def _op_extremum(self, intent: dict, func: str) -> QueryResult:
        """Shared handler for min/max operations to avoid duplication."""
        col = self._numeric_col(intent.column)
        if col is None:
            return QueryResult(
                operation=func, success=False,
                error=f"Cannot compute {func}: no numeric column found.",
            )
        filtered, used, skipped = self._apply_conditions(self.df, intent.conditions)
        series = pd.to_numeric(filtered[col], errors="coerce")
        if not series.dropna().empty:
            clean = series.dropna()
            val = getattr(clean, func)()
            idx_func = f"idx{func}"
            row = filtered.loc[[getattr(clean, idx_func)()]]
        else:
            val = None
            row = filtered.head(0)
        explanation = f"Computed df['{col}'].{func}() after filters -> {val}."
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation=func, success=True, scalar_result=val, table_result=row,
            columns_used=[col] + used,
            explanation=explanation,
        )

    def _op_min(self, intent: dict) -> QueryResult:
        return self._op_extremum(intent, "min")

    def _op_max(self, intent: dict) -> QueryResult:
        return self._op_extremum(intent, "max")

    def _op_filter(self, intent: dict) -> QueryResult:
        filtered, used, skipped = self._apply_conditions(self.df, intent.conditions)
        explanation = (
            f"Applied {len(intent.conditions)} filter condition(s) "
            f"-> {len(filtered)} matching rows."
        )
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation="filter", success=True, table_result=filtered, columns_used=used,
            scalar_result=len(filtered),
            explanation=explanation,
        )

    def _op_greater_than(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.column)
        val = intent.value
        conditions = [{"column": col, "op": "gt", "value": val}] + list(intent.conditions)
        filtered, used, skipped = self._apply_conditions(self.df, conditions)
        explanation = f"Filtered rows where {col} > {val} -> {len(filtered)} rows."
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation="greater_than", success=True, table_result=filtered,
            scalar_result=len(filtered), columns_used=[col] + used,
            explanation=explanation,
        )

    def _op_less_than(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.column)
        val = intent.value
        conditions = [{"column": col, "op": "lt", "value": val}] + list(intent.conditions)
        filtered, used, skipped = self._apply_conditions(self.df, conditions)
        explanation = f"Filtered rows where {col} < {val} -> {len(filtered)} rows."
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation="less_than", success=True, table_result=filtered,
            scalar_result=len(filtered), columns_used=[col] + used,
            explanation=explanation,
        )

    def _op_between(self, intent: dict) -> QueryResult:
        col = self._numeric_col(intent.column)
        val, val2 = intent.value, intent.value2
        conditions = [{"column": col, "op": "between", "value": val, "value2": val2}] + list(intent.conditions)
        filtered, used, skipped = self._apply_conditions(self.df, conditions)
        explanation = (
            f"Filtered rows where {col} is between {val} and {val2} "
            f"-> {len(filtered)} rows."
        )
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation="between", success=True, table_result=filtered,
            scalar_result=len(filtered), columns_used=[col] + used,
            explanation=explanation,
        )

    def _op_sort(self, intent: dict) -> QueryResult:
        col = self._resolve(intent.column, self.schema.primary_budget_col)
        if col is None:
            return QueryResult(
                operation="sort", success=False,
                error="Cannot sort: no column could be determined.",
            )
        ascending = bool((intent.ascending or False))
        filtered, used, skipped = self._apply_conditions(self.df, intent.conditions)
        sorted_df = filtered.sort_values(by=col, ascending=ascending)
        n = intent.n
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
        col = self._numeric_col(intent.column)
        n = int(intent.n or 5)
        filtered, used, _skipped = self._apply_conditions(self.df, intent.conditions)
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
            intent.group_by or intent.column,
            self.schema.location_col or self.schema.property_type_col,
        )
        agg_col = self._numeric_col(intent.agg_column)
        agg_func = (intent.agg_func or "mean").lower()
        filtered, used, _skipped = self._apply_conditions(self.df, intent.conditions)

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
        col = self._resolve(intent.column, self.schema.location_col)
        values = sorted(self.df[col].dropna().astype(str).unique().tolist()) if col else []
        return QueryResult(
            operation="unique", success=True,
            table_result=pd.DataFrame({col: values}) if col else None,
            scalar_result=values, columns_used=[col] if col else [],
            explanation=f"Computed df['{col}'].unique() -> {len(values)} distinct values.",
        )

    def _op_distinct_count(self, intent: dict) -> QueryResult:
        col = self._resolve(intent.column, self.schema.location_col)
        n = int(self.df[col].nunique(dropna=True)) if col else None
        return QueryResult(
            operation="distinct_count", success=True, scalar_result=n,
            columns_used=[col] if col else [],
            explanation=f"Computed df['{col}'].nunique() -> {n}.",
        )

    def _op_describe(self, intent: dict) -> QueryResult:
        col = self._resolve(intent.column)
        target = self.df[[col]] if col else self.df.select_dtypes(include="number")
        desc = target.describe()
        return QueryResult(
            operation="describe", success=True, table_result=desc,
            columns_used=[col] if col else [],
            explanation="Computed df.describe() summary statistics.",
        )

    def _op_list(self, intent: dict) -> QueryResult:
        filtered, used, skipped = self._apply_conditions(self.df, intent.conditions)
        n = intent.n
        if n:
            filtered = filtered.head(int(n))
        explanation = f"Listed {len(filtered)} matching row(s) after applying filters."
        if skipped:
            explanation += f" Skipped: {'; '.join(skipped)}."
        return QueryResult(
            operation="list", success=True, table_result=filtered, columns_used=used,
            scalar_result=len(filtered),
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Date filtering
    # ------------------------------------------------------------------
    def _op_date_filter(self, intent: dict) -> QueryResult:
        """Filter rows by date conditions (after/before/between dates)."""
        date_col = None
        for dc in self.schema.date_cols:
            if dc in self.df.columns:
                date_col = dc
                break
        if not date_col:
            return QueryResult(
                operation="date_filter", success=False,
                error="No date column detected in the dataset.",
            )

        op = (intent.date_op or "after").lower()
        date_val = intent.value
        date_val2 = intent.value2

        series = pd.to_datetime(self.df[date_col], errors="coerce")
        if op == "after" and date_val:
            mask = series >= pd.to_datetime(date_val, errors="coerce")
        elif op == "before" and date_val:
            mask = series <= pd.to_datetime(date_val, errors="coerce")
        elif op == "between" and date_val and date_val2:
            lo = pd.to_datetime(date_val, errors="coerce")
            hi = pd.to_datetime(date_val2, errors="coerce")
            mask = (series >= lo) & (series <= hi)
        elif op == "this_month":
            now = pd.Timestamp.now()
            mask = (series.dt.month == now.month) & (series.dt.year == now.year)
        else:
            mask = series.notna()

        filtered = self.df[mask]
        return QueryResult(
            operation="date_filter", success=True, table_result=filtered,
            scalar_result=len(filtered), columns_used=[date_col],
            explanation=f"Filtered by {date_col} ({op} '{date_val}') -> {len(filtered)} rows.",
        )

    # ------------------------------------------------------------------
    # Missing / NaN queries
    # ------------------------------------------------------------------
    def _op_missing(self, intent: dict) -> QueryResult:
        """Find rows with missing (NaN) values in a specific column."""
        col = self._resolve(intent.column)
        if not col:
            # Show all rows that have ANY missing value
            filtered = self.df[self.df.isna().any(axis=1)]
            return QueryResult(
                operation="missing", success=True, table_result=filtered,
                scalar_result=len(filtered), columns_used=[],
                explanation=f"Found {len(filtered)} rows with at least one missing value.",
            )
        filtered = self.df[self.df[col].isna()]
        return QueryResult(
            operation="missing", success=True, table_result=filtered,
            scalar_result=len(filtered), columns_used=[col],
            explanation=f"Found {len(filtered)} rows where '{col}' is missing/NaN.",
        )


# --------------------------------------------------------------------------
# Rule-based fallback intent parser (used if Gemini is unavailable / fails)
# --------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|cr|crores|k|thousand|million|m|billion|b|l)?", re.IGNORECASE)

_MULTIPLIERS = {
    "lakh": 100_000, "lac": 100_000, "lakhs": 100_000, "l": 100_000,
    "crore": 10_000_000, "cr": 10_000_000, "crores": 10_000_000,
    "k": 1_000, "thousand": 1_000,
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000,
}

# Word-number map for natural language number parsing ("one crore", "five hundred")
_WORD_NUMBERS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(_WORD_NUMBERS.keys()) + r")\s*(lakh|lac|lakhs|crore|cr|crores|k|thousand|million|m)?\b",
    re.IGNORECASE,
)


def _parse_number_with_unit(text: str) -> float | None:
    """Parse a number with optional unit from text.

    Supports both digit-based ("90 lakh") and word-based ("one crore")
    number expressions.
    """
    match = _NUMBER_RE.search(text)
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        return value * _MULTIPLIERS.get(unit, 1)

    # Try word-based numbers ("one crore", "five hundred")
    word_match = _WORD_NUMBER_RE.search(text)
    if word_match:
        value = _WORD_NUMBERS.get(word_match.group(1).lower(), 0)
        unit = (word_match.group(2) or "").lower()
        return value * _MULTIPLIERS.get(unit, 1)

    return None


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
) -> list[ConditionModel]:
    """Detect mentions of known categorical values (e.g. '2BHK', 'Pune')
    anywhere in the question and turn them into filter conditions, without
    hardcoding any specific column name."""
    conditions: list[ConditionModel] = []
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
                conditions.append(ConditionModel(column=col, op="contains", value=value))
                break  # only take the first matching value per column
    return conditions


def merge_follow_up_conditions(
    new_intent: RootIntentModel | SingleIntentModel, previous_conditions: list[ConditionModel]
) -> RootIntentModel | SingleIntentModel:
    """Merge conditions from a previous query into the new intent to
    support conversational follow-ups like 'only those above 90 lakhs'.

    Only merges if the new intent has no conditions of its own for a
    column that the previous query already filtered on.
    """
    if isinstance(new_intent, dict):
        new_intent = RootIntentModel.model_validate(new_intent)
    if previous_conditions and isinstance(previous_conditions[0], dict):
        previous_conditions = [ConditionModel.model_validate(c) for c in previous_conditions]

    if not previous_conditions:
        return new_intent

    existing = new_intent.conditions
    existing_cols = {c.column for c in existing if c.column}
    for cond in previous_conditions:
        if cond.column not in existing_cols:
            existing.append(cond)
    new_intent.conditions = existing
    return new_intent


# Date-related keywords used by the rule-based parser
# Multi-word phrases must be checked BEFORE single-word phrases
_DATE_MULTIWORD = re.compile(
    r"\b(this month|last month|this year|last year|recent|latest)",
    re.IGNORECASE,
)
_DATE_KEYWORDS = re.compile(
    r"\b(after|before|since|from|until)"
    r"\s*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)?\b",
    re.IGNORECASE,
)

_MONTH_MAP: dict[str, str] = {
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}


def validate_chain_intent(intent: RootIntentModel | dict) -> list[SingleIntentModel]:
    """Validate a multi-step intent and return the sanitized steps list.

    Raises ValueError if the structure is malformed.
    """
    if isinstance(intent, dict):
        intent = RootIntentModel.model_validate(intent)
        
    steps = intent.steps
    if not steps:
        raise ValueError("'steps' must be a non-empty list of operation objects.")
    if len(steps) > MAX_QUERY_STEPS:
        raise ValueError(
            f"Too many steps ({len(steps)}). "
            f"Maximum allowed is {MAX_QUERY_STEPS}."
        )
    validated: list[SingleIntentModel] = []
    for i, step in enumerate(steps):
        op = (step.operation or "").lower().strip()
        if op not in OPERATIONS:
            raise ValueError(
                f"Step {i + 1} has invalid operation '{op}'. "
                f"Allowed: {sorted(OPERATIONS)}"
            )
        validated.append(step)
    return validated


def _parse_single_intent(
    question: str, schema: DatasetSchema, df: pd.DataFrame | None = None
) -> SingleIntentModel:
    """Parse a single-operation intent from the question.

    This is the core keyword-based matching logic, extracted so that
    ``rule_based_intent`` can call it for each segment of a chained
    question while keeping the public API unchanged.
    """
    q = question.lower()
    budget_col = schema.primary_budget_col
    categorical_conditions = _match_categorical_conditions(q, schema, df)
    numbers = _extract_numbers_with_shared_units(q)

    # --- Missing / NaN queries ---
    if any(kw in q for kw in ("missing", "null", "nan", "empty", "blank", "unknown")):
        # Try to figure out which column
        col_hint = None
        for c in (schema.contact_col, schema.primary_budget_col, schema.name_col, schema.location_col, schema.status_col):
            if c and c.lower() in q:
                col_hint = c
                break
        # Also check common words
        if not col_hint:
            if any(w in q for w in ("phone", "contact", "mobile", "email")):
                col_hint = schema.contact_col
            elif any(w in q for w in ("budget", "price", "amount")):
                col_hint = schema.primary_budget_col
            elif any(w in q for w in ("name", "customer")):
                col_hint = schema.name_col
        return SingleIntentModel(operation="missing", column=col_hint)

    # --- Date queries ---
    # Check multi-word patterns first (e.g. "this month", "last month")
    multiword_match = _DATE_MULTIWORD.search(q)
    date_match = _DATE_KEYWORDS.search(q)

    if (multiword_match or date_match) and schema.date_cols:
        # Prefer multi-word match so "this month" is not parsed as just "this"
        if multiword_match:
            phrase = multiword_match.group(1).lower()
            if phrase == "this month":
                return SingleIntentModel(operation="date_filter", date_op="this_month", value=None)
            if phrase == "last month":

                now = datetime.date.today()
                first = now.replace(day=1)
                last_month_end = first - datetime.timedelta(days=1)
                last_month_start = last_month_end.replace(day=1)
                return SingleIntentModel(operation="date_filter", date_op="between",
                        value=str(last_month_start), value2=str(last_month_end))
            if phrase in ("this year",):

                year = datetime.date.today().year
                return SingleIntentModel(operation="date_filter", date_op="between",
                        value=f"January 1, {year}", value2=f"December 31, {year}")
            if phrase in ("recent", "latest"):
                return SingleIntentModel(operation="sort", column=schema.date_cols[0], ascending=False, n=10)

        if date_match:
            date_op_word = date_match.group(1).lower()
            month_word = (date_match.group(2) or "").lower()
            if month_word:
                full_month = _MONTH_MAP.get(month_word, month_word.capitalize())
                import datetime
                year = datetime.date.today().year
                if date_op_word in ("after", "since", "from"):
                    return SingleIntentModel(
                        operation="date_filter",
                        date_op="after",
                        value=f"{full_month} 1, {year}"
                    )
                elif date_op_word in ("before", "until"):
                    return SingleIntentModel(
                        operation="date_filter",
                        date_op="before",
                        value=f"{full_month} 1, {year}"
                    )

    if "unique" in q or "distinct" in q:
        col = schema.location_col if ("location" in q or "city" in q or "area" in q) else (
            schema.property_type_col if "type" in q else schema.location_col
        )
        if "how many" in q or "count" in q:
            return SingleIntentModel(operation="distinct_count", column=col)
        return SingleIntentModel(operation="unique", column=col)
    if "median" in q:
        return SingleIntentModel(operation="median", column=budget_col, conditions=categorical_conditions)
    if "average" in q or "avg" in q or "mean" in q:
        if ("by" in q or "each" in q or "per " in q) and (schema.location_col or schema.property_type_col):
            group_col = schema.location_col if ("city" in q or "location" in q) else (schema.property_type_col or schema.location_col)
            return SingleIntentModel(operation="groupby", group_by=group_col, agg_column=budget_col, agg_func="mean")
        return SingleIntentModel(operation="average", column=budget_col, conditions=categorical_conditions)
    if "sum" in q or "total" in q:
        return SingleIntentModel(operation="sum", column=budget_col, conditions=categorical_conditions)
    if "highest" in q or "max" in q or "maximum" in q:
        if ("by" in q or "which" in q) and (schema.location_col or schema.property_type_col):
            group_col = schema.location_col or schema.property_type_col
            return SingleIntentModel(operation="groupby", group_by=group_col, agg_column=budget_col, agg_func="mean")
        return SingleIntentModel(operation="max", column=budget_col, conditions=categorical_conditions)
    if "lowest" in q or "min" in q or "minimum" in q or "cheapest" in q:
        return SingleIntentModel(operation="min", column=budget_col, conditions=categorical_conditions)
    if "between" in q and len(numbers) >= 2:
        return SingleIntentModel(
            operation="between",
            column=budget_col,
            value=min(numbers),
            value2=max(numbers),
            conditions=categorical_conditions
        )
    if ("above" in q or "greater" in q or "more than" in q or "over" in q) and numbers:
        return SingleIntentModel(
            operation="greater_than",
            column=budget_col,
            value=numbers[0],
            conditions=categorical_conditions
        )
    if ("below" in q or "less than" in q or "under" in q) and numbers:
        return SingleIntentModel(
            operation="less_than",
            column=budget_col,
            value=numbers[0],
            conditions=categorical_conditions
        )
    if "top" in q:
        n_match = re.search(r"top\s+(\d+)", q)
        return SingleIntentModel(operation="topn", column=budget_col, n=int(n_match.group(1)) if n_match else 5, conditions=categorical_conditions)
    if "bottom" in q:
        n_match = re.search(r"bottom\s+(\d+)", q)
        return SingleIntentModel(operation="bottomn", column=budget_col, n=int(n_match.group(1)) if n_match else 5, conditions=categorical_conditions)
    if "group" in q or "breakdown" in q or "distribution" in q or "each" in q or "per" in q or "most" in q or "least" in q or "popular" in q or re.search(r"count\b.*?\bby\b", q):
        group_col = schema.status_col or schema.location_col or schema.property_type_col
        # Try to infer a specific group col from the question
        if "location" in q or "city" in q or "area" in q:
            group_col = schema.location_col
        elif "type" in q or "property" in q:
            group_col = schema.property_type_col
        elif "status" in q:
            group_col = schema.status_col
            
        return SingleIntentModel(operation="groupby", group_by=group_col, agg_column=None, agg_func="count")
    if "list" in q or "show" in q or "details" in q:
        return SingleIntentModel(operation="list", conditions=categorical_conditions)
    if "sort" in q or "order" in q:
        return SingleIntentModel(operation="sort", column=budget_col, ascending=("asc" in q or "lowest" in q), conditions=categorical_conditions)
    return SingleIntentModel(operation="count", conditions=categorical_conditions)


# Regex for detecting chain boundaries in natural-language questions.
# Matches " then ", " and then ", or ", then " as step separators.
_CHAIN_SPLIT_RE = re.compile(r",?\s+(?:and\s+)?then\s+", re.IGNORECASE)


def rule_based_intent(
    question: str, schema: DatasetSchema, df: pd.DataFrame | None = None
) -> RootIntentModel:
    """A conservative keyword-based fallback so the app still works
    without an LLM (or if the Gemini call fails / quota is exceeded).

    If the question contains chain markers (e.g. "… then sort by …"),
    each segment is parsed independently and returned as
    ``RootIntentModel(steps=[intent1, intent2, …])``.  Otherwise a single intent
    is returned.
    """
    parts = _CHAIN_SPLIT_RE.split(question)
    if len(parts) > 1:
        steps = [
            _parse_single_intent(part.strip(), schema, df)
            for part in parts
            if part.strip()
        ]
        if len(steps) > 1:
            return RootIntentModel(steps=steps[:MAX_QUERY_STEPS])
    
    single = _parse_single_intent(question, schema, df)
    return RootIntentModel(**single.model_dump())
