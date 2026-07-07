"""
utils.py
--------
Data loading, dataset profiling, and DYNAMIC schema detection.

The whole point of this module is that the app should never assume a
fixed column name like "City" or "Budget (INR)". Instead it inspects
whatever Excel file the user uploads, classifies each column into a
semantic "role" (name, budget/numeric-money, location, category,
status, contact, date/period, other-numeric, other-categorical) and
hands that mapping to the query engine.

This lets the same app work on the sample "Pune real-estate leads"
sheet as well as any other customer-style spreadsheet with different
column names (e.g. "Price", "City", "Lead Status", "Phone").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Keyword dictionaries used for semantic column classification.
# These are intentionally broad so the app generalizes to similar datasets.
# --------------------------------------------------------------------------
NAME_KEYWORDS = ["name", "customer", "client", "lead name", "full name"]
BUDGET_KEYWORDS = [
    "budget", "price", "amount", "cost", "value", "salary", "income",
    "revenue", "fee", "rent", "expense", "spend",
]
LOCATION_KEYWORDS = ["location", "city", "area", "place", "region", "locality", "address"]
PROPERTY_TYPE_KEYWORDS = ["property type", "type", "category", "bhk", "unit type", "segment"]
STATUS_KEYWORDS = ["status", "stage", "call status", "lead status", "outcome"]
CONTACT_KEYWORDS = ["contact", "phone", "mobile", "number", "email", "whatsapp"]
DATE_KEYWORDS = ["date", "time", "possession", "created", "updated", "connected", "timestamp"]
ID_KEYWORDS = ["id", "uuid", "code", "ref"]


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing: int
    missing_pct: float
    unique_count: int
    role: str  # semantic role assigned
    sample_values: list = field(default_factory=list)


@dataclass
class DatasetSchema:
    """Holds the semantic mapping discovered for the uploaded dataset."""

    name_col: str | None = None
    budget_cols: list[str] = field(default_factory=list)
    primary_budget_col: str | None = None
    location_col: str | None = None
    property_type_col: str | None = None
    status_col: str | None = None
    contact_col: str | None = None
    date_cols: list[str] = field(default_factory=list)
    id_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    all_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_col": self.name_col,
            "budget_cols": self.budget_cols,
            "primary_budget_col": self.primary_budget_col,
            "location_col": self.location_col,
            "property_type_col": self.property_type_col,
            "status_col": self.status_col,
            "contact_col": self.contact_col,
            "date_cols": self.date_cols,
            "id_cols": self.id_cols,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "all_columns": self.all_columns,
        }


def _matches_any(col_name: str, keywords: list[str]) -> bool:
    lowered = col_name.lower()
    return any(kw in lowered for kw in keywords)


def load_dataframe(file) -> pd.DataFrame:
    """Load an uploaded Excel file (path or file-like object) into a DataFrame.

    Raises a ValueError with a friendly message on failure so the UI layer
    can surface it cleanly instead of a raw traceback.
    """
    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001 - we want a friendly wrapper
        raise ValueError(f"Could not read the Excel file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded Excel file has no data rows.")

    # Drop fully-empty columns/rows that sometimes appear from Excel exports
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Normalize column names: strip whitespace
    df.columns = [str(c).strip() for c in df.columns]

    return df


def detect_schema(df: pd.DataFrame) -> DatasetSchema:
    """Inspect the dataframe and classify each column into a semantic role."""
    schema = DatasetSchema(all_columns=list(df.columns))

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    for col in df.columns:
        # Name column: first text column whose values look like proper names
        if schema.name_col is None and _matches_any(col, NAME_KEYWORDS):
            schema.name_col = col
            continue

        if _matches_any(col, ID_KEYWORDS) and col not in schema.id_cols:
            schema.id_cols.append(col)
            continue

        if _matches_any(col, CONTACT_KEYWORDS):
            if schema.contact_col is None:
                schema.contact_col = col
            continue

        if _matches_any(col, DATE_KEYWORDS):
            schema.date_cols.append(col)
            continue

        if col in numeric_cols and _matches_any(col, BUDGET_KEYWORDS):
            schema.budget_cols.append(col)
            continue

        if _matches_any(col, LOCATION_KEYWORDS):
            if schema.location_col is None:
                schema.location_col = col
            continue

        if _matches_any(col, PROPERTY_TYPE_KEYWORDS):
            if schema.property_type_col is None:
                schema.property_type_col = col
            continue

        if _matches_any(col, STATUS_KEYWORDS):
            if schema.status_col is None:
                schema.status_col = col
            continue

    # Fallbacks: if no explicit "budget" style column found, pick the numeric
    # column with the largest average magnitude (money-like columns tend to
    # dominate other numeric columns such as counts or small IDs).
    if not schema.budget_cols:
        candidate_numeric = [c for c in numeric_cols if c not in schema.id_cols]
        if candidate_numeric:
            means = {c: pd.to_numeric(df[c], errors="coerce").mean() for c in candidate_numeric}
            best = max(means, key=lambda k: (means[k] if means[k] == means[k] else -1))
            schema.budget_cols.append(best)

    schema.primary_budget_col = schema.budget_cols[0] if schema.budget_cols else None

    # Fallback for name column: first non-numeric column with high uniqueness
    if schema.name_col is None:
        for col in non_numeric_cols:
            if col in schema.date_cols or col in schema.id_cols:
                continue
            uniq_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if uniq_ratio > 0.8:
                schema.name_col = col
                break

    # Final classification buckets used by the profiler / UI
    schema.numeric_cols = numeric_cols
    used_cols = {
        schema.name_col,
        schema.contact_col,
        schema.location_col,
        schema.property_type_col,
        schema.status_col,
        *schema.date_cols,
        *schema.id_cols,
        *schema.budget_cols,
    }
    schema.categorical_cols = [
        c for c in non_numeric_cols if c not in used_cols and c is not None
    ]
    # Location / property-type / status are categorical too, keep them visible
    for c in [schema.location_col, schema.property_type_col, schema.status_col]:
        if c and c not in schema.categorical_cols:
            schema.categorical_cols.append(c)

    return schema


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the dataset-wide statistics shown in the UI's profiling panel."""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    col_profiles: list[ColumnProfile] = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        col_profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(df[col].dtype),
                missing=missing,
                missing_pct=round(100 * missing / max(len(df), 1), 2),
                unique_count=int(df[col].nunique(dropna=True)),
                role="numeric" if col in numeric_cols else "categorical",
                sample_values=df[col].dropna().astype(str).unique()[:5].tolist(),
            )
        )

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_profiles": col_profiles,
    }


def rule_based_insights(df: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """Generate simple, 100% pandas-derived insight strings (no AI).

    These are always safe to show even if the Gemini API is unavailable,
    and they are also fed to Gemini as ground truth so it never has to
    guess a number on its own.
    """
    insights: list[str] = []

    if schema.primary_budget_col and schema.primary_budget_col in df.columns:
        col = schema.primary_budget_col
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            insights.append(
                f"Average {col} is {series.mean():,.0f} "
                f"(min {series.min():,.0f}, max {series.max():,.0f})."
            )

    if schema.property_type_col and schema.property_type_col in df.columns:
        counts = df[schema.property_type_col].value_counts()
        if not counts.empty:
            top = counts.idxmax()
            pct = 100 * counts.max() / counts.sum()
            insights.append(
                f"The most common {schema.property_type_col} is '{top}' "
                f"({pct:.1f}% of records)."
            )

    if schema.location_col and schema.location_col in df.columns:
        counts = df[schema.location_col].value_counts()
        if not counts.empty:
            top = counts.idxmax()
            insights.append(
                f"'{top}' is the most frequent {schema.location_col} "
                f"with {counts.max()} records."
            )

    if schema.status_col and schema.status_col in df.columns:
        counts = df[schema.status_col].value_counts()
        if not counts.empty:
            top = counts.idxmax()
            pct = 100 * counts.max() / counts.sum()
            insights.append(
                f"'{top}' is the most common {schema.status_col} "
                f"({pct:.1f}% of records)."
            )

    dup = int(df.duplicated().sum())
    if dup > 0:
        insights.append(f"There are {dup} duplicate rows in the dataset.")

    missing = int(df.isna().sum().sum())
    if missing > 0:
        insights.append(f"The dataset has {missing} missing values across all columns.")
    else:
        insights.append("The dataset has no missing values.")

    return insights


def resolve_column(user_text: str, columns: list[str]) -> str | None:
    """Best-effort fuzzy match of a free-text column reference to a real column.

    Used as a fallback when Gemini's structured intent names a column that
    doesn't exactly match the dataframe (e.g. 'budget' -> 'Budget (INR)').
    """
    if not user_text:
        return None
    text = user_text.strip().lower()

    for col in columns:
        if col.strip().lower() == text:
            return col

    for col in columns:
        cl = col.strip().lower()
        if text in cl or cl in text:
            return col

    # token overlap fallback
    text_tokens = set(re.findall(r"[a-z0-9]+", text))
    best_col, best_score = None, 0
    for col in columns:
        col_tokens = set(re.findall(r"[a-z0-9]+", col.lower()))
        score = len(text_tokens & col_tokens)
        if score > best_score:
            best_col, best_score = col, score
    return best_col if best_score > 0 else None
