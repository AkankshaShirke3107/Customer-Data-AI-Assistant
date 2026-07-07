"""
charts.py
---------
Automatic chart-type selection built on Plotly. Given a QueryResult
(from query_engine.py) and the schema, decide whether a visualization
would help and, if so, build the right Plotly figure.

All charts use a dark-theme-compatible palette to match the premium UI.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import DEFAULT_CHART_MARGIN
from utils import DatasetSchema

# ---------------------------------------------------------------------------
# Dark theme palette — matches app.py's #09090B background
# ---------------------------------------------------------------------------
_DARK_FONT = dict(family="Inter, sans-serif", color="#A1A1AA", size=12)
_DARK_TITLE_FONT = dict(family="Inter, sans-serif", color="#FAFAFA", size=14)
_DARK_GRIDCOLOR = "rgba(255,255,255,.04)"
_DARK_LINECOLOR = "rgba(255,255,255,.06)"

# Curated color sequences for dark backgrounds
_ACCENT_SEQUENCE = [
    "#3B82F6",  # blue
    "#8B5CF6",  # violet
    "#22C55E",  # green
    "#F59E0B",  # amber
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#F97316",  # orange
    "#A78BFA",  # light-violet
]

_LAYOUT_DEFAULTS = dict(
    margin=DEFAULT_CHART_MARGIN,
    font=_DARK_FONT,
    title_font=_DARK_TITLE_FONT,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        gridcolor=_DARK_GRIDCOLOR,
        linecolor=_DARK_LINECOLOR,
        zerolinecolor=_DARK_GRIDCOLOR,
    ),
    yaxis=dict(
        gridcolor=_DARK_GRIDCOLOR,
        linecolor=_DARK_LINECOLOR,
        zerolinecolor=_DARK_GRIDCOLOR,
    ),
    colorway=_ACCENT_SEQUENCE,
    legend=dict(font=dict(color="#A1A1AA", size=11)),
    hoverlabel=dict(
        bgcolor="#18181B",
        bordercolor="rgba(255,255,255,.08)",
        font=dict(color="#FAFAFA", family="Inter, sans-serif", size=12),
    ),
)

CHART_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "responsive": True,
}


def _apply_layout(fig: go.Figure, **extra) -> go.Figure:
    """Apply dark-theme layout defaults to any figure."""
    fig.update_layout(**_LAYOUT_DEFAULTS, **extra)
    return fig


def _is_small_table(df: pd.DataFrame, max_rows: int = 60) -> bool:
    return df is not None and 1 < len(df) <= max_rows


# ---------------------------------------------------------------------------
# Auto-visualization for query results
# ---------------------------------------------------------------------------
def auto_visualize(
    operation: str,
    table: pd.DataFrame | None,
    schema: DatasetSchema,
    columns_used: list[str],
) -> go.Figure | None:
    """Return a Plotly Figure, or None if no sensible chart applies."""
    if table is None or table.empty:
        return None

    numeric_cols = list(table.select_dtypes(include="number").columns)
    non_numeric_cols = [c for c in table.columns if c not in numeric_cols]

    try:
        if operation == "groupby" and len(table.columns) == 2:
            cat_col, val_col = table.columns[0], table.columns[1]
            fig = px.bar(
                table.sort_values(by=val_col, ascending=False),
                x=cat_col, y=val_col, text_auto=".2s",
                title=f"{val_col} by {cat_col}",
                color=val_col, color_continuous_scale="Viridis",
            )
            fig.update_coloraxes(colorbar=dict(tickfont=dict(color="#71717A")))
            return _apply_layout(fig)

        if operation in {"topn", "bottomn", "sort"} and _is_small_table(table):
            name_col = (
                schema.name_col if schema.name_col in table.columns
                else non_numeric_cols[0] if non_numeric_cols
                else table.columns[0]
            )
            val_col = None
            for c in columns_used:
                if c in numeric_cols:
                    val_col = c
                    break
            val_col = val_col or (numeric_cols[0] if numeric_cols else None)
            if val_col:
                fig = px.bar(
                    table, x=name_col, y=val_col,
                    title=f"{val_col} by {name_col}",
                    color_discrete_sequence=["#3B82F6"],
                )
                return _apply_layout(fig, xaxis_tickangle=-40)

        if operation in {"filter", "list", "greater_than", "less_than", "between"} and _is_small_table(table):
            cat_col = (
                schema.property_type_col if schema.property_type_col in table.columns
                else schema.location_col if schema.location_col in table.columns
                else None
            )
            if cat_col:
                counts = table[cat_col].value_counts().reset_index()
                counts.columns = [cat_col, "count"]
                fig = px.pie(
                    counts, names=cat_col, values="count",
                    title=f"Distribution of {cat_col} in results", hole=0.4,
                    color_discrete_sequence=_ACCENT_SEQUENCE,
                )
                return _apply_layout(fig)
            if numeric_cols:
                fig = px.histogram(
                    table, x=numeric_cols[0], nbins=20,
                    title=f"Distribution of {numeric_cols[0]}",
                    color_discrete_sequence=["#3B82F6"],
                )
                return _apply_layout(fig)

        if operation == "unique" and table is not None and len(table.columns) == 1:
            col = table.columns[0]
            fig = px.bar(
                table, x=col, title=f"Unique values of {col}",
                color_discrete_sequence=["#8B5CF6"],
            )
            return _apply_layout(fig, xaxis_tickangle=-40)

        if operation == "describe":
            return None

    except Exception:  # noqa: BLE001 – visualization is best-effort
        return None

    return None


# ---------------------------------------------------------------------------
# Profile overview charts (displayed after file upload)
# ---------------------------------------------------------------------------
def profile_overview_charts(
    df: pd.DataFrame, schema: DatasetSchema
) -> dict[str, go.Figure]:
    """Charts shown right after upload, before any question is asked."""
    figs: dict[str, go.Figure] = {}

    if schema.primary_budget_col and schema.primary_budget_col in df.columns:
        fig = px.histogram(
            df, x=schema.primary_budget_col, nbins=25,
            title=f"Distribution of {schema.primary_budget_col}",
            color_discrete_sequence=["#3B82F6"],
        )
        figs["budget_hist"] = _apply_layout(fig)

    if schema.location_col and schema.location_col in df.columns:
        counts = df[schema.location_col].value_counts().reset_index()
        counts.columns = [schema.location_col, "count"]
        fig = px.bar(
            counts, x=schema.location_col, y="count",
            title=f"Records by {schema.location_col}",
            color_discrete_sequence=["#8B5CF6"],
        )
        figs["location_bar"] = _apply_layout(fig, xaxis_tickangle=-40)

    if schema.property_type_col and schema.property_type_col in df.columns:
        counts = df[schema.property_type_col].value_counts().reset_index()
        counts.columns = [schema.property_type_col, "count"]
        fig = px.pie(
            counts, names=schema.property_type_col, values="count",
            title=f"Share by {schema.property_type_col}", hole=0.4,
            color_discrete_sequence=_ACCENT_SEQUENCE,
        )
        figs["property_type_pie"] = _apply_layout(fig)

    if schema.status_col and schema.status_col in df.columns:
        counts = df[schema.status_col].value_counts().reset_index()
        counts.columns = [schema.status_col, "count"]
        fig = px.bar(
            counts, x=schema.status_col, y="count",
            title=f"{schema.status_col} breakdown",
            color_discrete_sequence=["#22C55E"],
        )
        figs["status_bar"] = _apply_layout(fig, xaxis_tickangle=-30)

    if (
        schema.primary_budget_col
        and schema.location_col
        and schema.primary_budget_col in df.columns
        and schema.location_col in df.columns
    ):
        fig = px.box(
            df, x=schema.location_col, y=schema.primary_budget_col,
            title=f"{schema.primary_budget_col} spread by {schema.location_col}",
            color=schema.location_col,
            color_discrete_sequence=_ACCENT_SEQUENCE,
        )
        figs["budget_by_location_box"] = _apply_layout(
            fig, xaxis_tickangle=-40, showlegend=False,
        )

    return figs
