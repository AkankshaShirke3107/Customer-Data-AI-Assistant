"""
charts.py
---------
Automatic chart-type selection built on Plotly. Given a QueryResult
(from query_engine.py) and the schema, decide whether a visualization
would help and, if so, build the right Plotly figure.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import DatasetSchema


def _is_small_table(df: pd.DataFrame, max_rows: int = 60) -> bool:
    return df is not None and 1 < len(df) <= max_rows


def auto_visualize(operation: str, table: pd.DataFrame | None, schema: DatasetSchema, columns_used: list[str]):
    """Return a plotly Figure, or None if no sensible chart applies."""
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
                color=val_col, color_continuous_scale="Tealgrn",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            return fig

        if operation in {"topn", "bottomn", "sort"} and _is_small_table(table):
            name_col = schema.name_col if schema.name_col in table.columns else non_numeric_cols[0] if non_numeric_cols else table.columns[0]
            val_col = None
            for c in columns_used:
                if c in numeric_cols:
                    val_col = c
                    break
            val_col = val_col or (numeric_cols[0] if numeric_cols else None)
            if val_col:
                fig = px.bar(
                    table, x=name_col, y=val_col, title=f"{val_col} by {name_col}",
                    color=val_col, color_continuous_scale="Blues",
                )
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-40)
                return fig

        if operation in {"filter", "list", "greater_than", "less_than", "between"} and _is_small_table(table):
            cat_col = schema.property_type_col if schema.property_type_col in table.columns else (
                schema.location_col if schema.location_col in table.columns else None
            )
            if cat_col:
                counts = table[cat_col].value_counts().reset_index()
                counts.columns = [cat_col, "count"]
                fig = px.pie(counts, names=cat_col, values="count", title=f"Distribution of {cat_col} in results", hole=0.35)
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                return fig
            if numeric_cols:
                fig = px.histogram(table, x=numeric_cols[0], nbins=20, title=f"Distribution of {numeric_cols[0]}")
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                return fig

        if operation == "unique" and table is not None and len(table.columns) == 1:
            col = table.columns[0]
            fig = px.bar(table, x=col, title=f"Unique values of {col}")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-40)
            return fig

        if operation == "describe":
            return None  # a styled table is clearer than a chart here

    except Exception:  # noqa: BLE001 - visualization is best-effort only
        return None

    return None


def profile_overview_charts(df: pd.DataFrame, schema: DatasetSchema) -> dict[str, go.Figure]:
    """Charts shown right after upload, before any question is asked."""
    figs: dict[str, go.Figure] = {}

    if schema.primary_budget_col and schema.primary_budget_col in df.columns:
        fig = px.histogram(
            df, x=schema.primary_budget_col, nbins=25,
            title=f"Distribution of {schema.primary_budget_col}",
            color_discrete_sequence=["#6366F1"],
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        figs["budget_hist"] = fig

    if schema.location_col and schema.location_col in df.columns:
        counts = df[schema.location_col].value_counts().reset_index()
        counts.columns = [schema.location_col, "count"]
        fig = px.bar(
            counts, x=schema.location_col, y="count",
            title=f"Records by {schema.location_col}", color="count",
            color_continuous_scale="Purp",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-40)
        figs["location_bar"] = fig

    if schema.property_type_col and schema.property_type_col in df.columns:
        counts = df[schema.property_type_col].value_counts().reset_index()
        counts.columns = [schema.property_type_col, "count"]
        fig = px.pie(
            counts, names=schema.property_type_col, values="count",
            title=f"Share by {schema.property_type_col}", hole=0.4,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        figs["property_type_pie"] = fig

    if schema.status_col and schema.status_col in df.columns:
        counts = df[schema.status_col].value_counts().reset_index()
        counts.columns = [schema.status_col, "count"]
        fig = px.bar(
            counts, x=schema.status_col, y="count", title=f"{schema.status_col} breakdown",
            color="count", color_continuous_scale="Sunset",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-30)
        figs["status_bar"] = fig

    if schema.primary_budget_col and schema.location_col and schema.primary_budget_col in df.columns and schema.location_col in df.columns:
        fig = px.box(
            df, x=schema.location_col, y=schema.primary_budget_col,
            title=f"{schema.primary_budget_col} spread by {schema.location_col}",
            color=schema.location_col,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-40, showlegend=False)
        figs["budget_by_location_box"] = fig

    return figs
