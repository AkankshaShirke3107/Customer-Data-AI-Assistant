"""
app.py
------
Customer Data AI Assistant — "Chat with your Excel data using Natural
Language."

Streamlit front-end that ties together:
  config.py        -> centralized configuration
  utils.py         -> loading + dynamic schema detection + profiling
  query_engine.py  -> deterministic pandas execution (no hallucination)
  gemini_helper.py -> Gemini for intent understanding + summarization only
  charts.py        -> automatic Plotly visualizations

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import charts
import gemini_helper
from charts import CHART_CONFIG
from config import (
    CACHE_TTL_SECONDS,
    DATA_PREVIEW_ROWS,
    LOG_FORMAT,
    LOG_LEVEL,
    MAX_CHAT_HISTORY,
    TABLE_DISPLAY_ROWS,
)
from query_engine import QueryEngine, merge_follow_up_conditions, rule_based_intent
from utils import (
    DatasetSchema,
    detect_schema,
    load_dataframe,
    profile_dataset,
    rule_based_insights,
    validate_upload,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Customer Data AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# THEME / CSS — Premium SaaS appearance
# --------------------------------------------------------------------------
def inject_css(dark_mode: bool) -> None:
    """Inject a premium CSS theme with smooth transitions and modern typography."""
    if dark_mode:
        bg = "#0F1116"
        panel = "#1A1D27"
        text = "#F2F3F7"
        sub = "#9AA1B5"
        accent = "#7C8CFF"
        border = "#2A2E3D"
        card_shadow = "rgba(0,0,0,0.3)"
        hover_lift = "rgba(124,140,255,0.08)"
    else:
        bg = "#F7F8FC"
        panel = "#FFFFFF"
        text = "#1A1D27"
        sub = "#666E82"
        accent = "#6366F1"
        border = "#E7E9F3"
        card_shadow = "rgba(0,0,0,0.06)"
        hover_lift = "rgba(99,102,241,0.06)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        *, html, body, .stApp {{ font-family: 'Inter', sans-serif !important; }}
        .stApp {{ background-color: {bg}; color: {text}; }}
        section[data-testid="stSidebar"] {{
            background-color: {panel}; border-right: 1px solid {border};
        }}

        /* ---------- Metric cards ---------- */
        .metric-card {{
            background: {panel}; border: 1px solid {border}; border-radius: 16px;
            padding: 20px 22px; text-align: left;
            box-shadow: 0 2px 8px {card_shadow};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px {card_shadow};
        }}
        .metric-value {{ font-size: 30px; font-weight: 800; color: {accent}; }}
        .metric-label {{ font-size: 13px; color: {sub}; margin-top: 4px; font-weight: 500; }}

        /* ---------- App header ---------- */
        .app-header {{
            padding: 28px 30px; border-radius: 20px; margin-bottom: 22px;
            background: linear-gradient(135deg, {accent}22, {accent}08);
            border: 1px solid {border};
        }}
        .app-title {{
            font-size: 32px; font-weight: 800; color: {text}; margin-bottom: 4px;
            letter-spacing: -0.5px;
        }}
        .app-tagline {{ font-size: 15px; color: {sub}; font-weight: 500; }}

        /* ---------- Insight pills ---------- */
        .insight-pill {{
            display: block; background: {panel}; border: 1px solid {border};
            border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;
            font-size: 14px; color: {text}; font-weight: 500;
            transition: background 0.2s ease;
        }}
        .insight-pill:hover {{ background: {hover_lift}; }}

        /* ---------- Chat bubbles ---------- */
        .chat-bubble-user {{
            background: linear-gradient(135deg, {accent}, {accent}DD);
            color: white; padding: 12px 18px; border-radius: 18px 18px 4px 18px;
            display: inline-block; margin: 8px 0; max-width: 85%;
            font-weight: 500; box-shadow: 0 2px 8px {card_shadow};
        }}
        .chat-bubble-ai {{
            background: {panel}; border: 1px solid {border}; color: {text};
            padding: 12px 18px; border-radius: 18px 18px 18px 4px;
            display: inline-block; margin: 8px 0; max-width: 85%;
            box-shadow: 0 2px 8px {card_shadow};
        }}

        /* ---------- Confidence badge ---------- */
        .confidence-badge {{
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            font-size: 12px; font-weight: 600;
            background: {accent}18; color: {accent};
        }}
        .confidence-badge-high {{
            background: #10B98118; color: #10B981;
        }}

        /* ---------- Pipeline step ---------- */
        .pipeline-step {{
            display: flex; align-items: center; gap: 10px;
            padding: 8px 0; font-size: 13px; color: {sub};
        }}
        .pipeline-icon {{
            width: 28px; height: 28px; border-radius: 8px;
            background: {accent}15; color: {accent};
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; font-weight: 700; flex-shrink: 0;
        }}

        /* ---------- Suggested query cards ---------- */
        .sugg-card {{
            background: {panel}; border: 1px solid {border}; border-radius: 12px;
            padding: 0; text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
        }}
        .sugg-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px {card_shadow};
        }}

        /* ---------- Section headers ---------- */
        .section-header {{
            font-size: 20px; font-weight: 700; color: {text};
            margin: 24px 0 12px 0; letter-spacing: -0.3px;
        }}

        /* ---------- Buttons ---------- */
        div.stButton > button {{
            border-radius: 10px; border: 1px solid {border}; font-weight: 600;
            transition: all 0.2s ease;
        }}
        div.stButton > button:hover {{
            border-color: {accent}; color: {accent};
        }}

        /* ---------- Expander styling ---------- */
        .streamlit-expanderHeader {{
            font-weight: 600 !important;
        }}

        /* ---------- Footer ---------- */
        .app-footer {{
            text-align: center; padding: 20px 0; font-size: 12px;
            color: {sub}; border-top: 1px solid {border}; margin-top: 40px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# CACHED HELPERS
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load_dataframe(file_bytes: bytes) -> pd.DataFrame:
    return load_dataframe(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_profile(_df_hash: str, df: pd.DataFrame) -> dict:
    return profile_dataset(df)


@st.cache_data(show_spinner=False)
def cached_schema(_df_hash: str, df: pd.DataFrame):
    return detect_schema(df)


@st.cache_data(show_spinner=False)
def cached_rule_insights(_df_hash: str, df: pd.DataFrame, _schema) -> list[str]:
    return rule_based_insights(df, _schema)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_gemini_insights(stats_key: str, stats_lines: list[str]) -> list[str]:
    return gemini_helper.generate_insights(stats_lines)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_intent(question: str, columns: list[str], schema_dict: dict) -> dict | None:
    try:
        return gemini_helper.understand_intent(question, columns, schema_dict)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_summary(question: str, operation: str, payload: dict) -> str | None:
    try:
        return gemini_helper.summarize_result(question, operation, payload)
    except Exception:
        return None


def _df_hash(df: pd.DataFrame) -> str:
    """Cheap content hash for cache keying without hashing the full DataFrame."""
    return f"{len(df)}_{df.shape[1]}_{hash(tuple(df.columns))}_{hash(tuple(df.dtypes))}"


def build_result_payload(result) -> dict:
    payload = {"operation": result.operation, "scalar_result": result.scalar_result}
    if result.table_result is not None and not result.table_result.empty:
        payload["table_preview"] = result.table_result.head(15).to_dict(orient="records")
        payload["row_count"] = len(result.table_result)
    return payload


def fallback_summary(question: str, result) -> str:
    """Deterministic, non-AI phrasing used if Gemini summarization fails."""
    if not result.success:
        return f"⚠️ {result.error}"
    if result.scalar_result is not None and (
        result.table_result is None
        or result.operation in {"count", "sum", "average", "min", "max", "distinct_count"}
    ):
        val = result.scalar_result
        if isinstance(val, float):
            val = round(val, 2)
        return f"Result for **{result.operation}**: **{val}**"
    if result.table_result is not None:
        return f"Found **{len(result.table_result)}** matching row(s)."
    return "No result could be computed for this question."


def _trim_chat_history() -> None:
    """Evict oldest entries if chat history exceeds the configured max."""
    history = st.session_state.get("chat_history", [])
    if len(history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history = history[-MAX_CHAT_HISTORY:]


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark mode", value=True)
    st.divider()

    st.markdown("## 📁 Data Source")
    uploaded_file = st.file_uploader(
        "Upload your Excel file (.xlsx)", type=["xlsx", "xls"]
    )
    use_sample = False
    if uploaded_file is None:
        use_sample = st.checkbox(
            "Use bundled sample dataset (Pune real-estate leads)", value=True
        )

    st.divider()
    st.markdown("## 🔑 Gemini API")
    if gemini_helper.is_configured():
        st.success("✅ Gemini API key detected.")
    else:
        st.warning(
            "No GEMINI_API_KEY found. The app will still work using a "
            "rule-based fallback engine, but AI summaries/insights will "
            "be simplified."
        )
        manual_key = st.text_input(
            "Paste a Gemini API key for this session (optional)",
            type="password",
        )
        if manual_key:
            # Store in session state for isolation; also set env so
            # the Gemini helper picks it up within this process.
            st.session_state["_gemini_key"] = manual_key
            os.environ["GEMINI_API_KEY"] = manual_key
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("last_conditions", None)
        st.rerun()

    # About section
    st.divider()
    st.markdown("## ℹ️ About")
    st.caption(
        "**Customer Data AI Assistant** v2.0\n\n"
        "Built with Streamlit · Pandas · Plotly · Google Gemini\n\n"
        "Every number in every answer is computed by pandas. "
        "Gemini only understands your question and phrases the result."
    )

inject_css(dark_mode)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">📊 Customer Data AI Assistant</div>
        <div class="app-tagline">Chat with your Excel data using Natural Language — zero hallucination guaranteed.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
df = None
load_error = None

try:
    if uploaded_file is not None:
        validate_upload(uploaded_file)
        df = cached_load_dataframe(uploaded_file.getvalue())
    elif use_sample:
        sample_path = os.path.join(
            os.path.dirname(__file__), "data", "sample_leads.xlsx"
        )
        with open(sample_path, "rb") as f:
            df = cached_load_dataframe(f.read())
except ValueError as exc:
    load_error = str(exc)

if load_error:
    st.error(f"❌ {load_error}")
    st.stop()

if df is None:
    # Premium empty state
    st.markdown("---")
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown(
            """
            <div style="text-align:center; padding: 60px 20px;">
                <div style="font-size: 64px; margin-bottom: 16px;">📂</div>
                <div style="font-size: 22px; font-weight: 700; margin-bottom: 8px;">
                    No dataset loaded
                </div>
                <div style="font-size: 15px; color: #9AA1B5; max-width: 400px; margin: 0 auto;">
                    Upload an Excel file from the sidebar or check
                    <b>"Use bundled sample dataset"</b> to explore the app instantly.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

df_h = _df_hash(df)
schema = cached_schema(df_h, df)
profile = cached_profile(df_h, df)

# --------------------------------------------------------------------------
# DATASET OVERVIEW — Metric cards with hover effects
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("📊", "Rows", profile["rows"]),
    ("📐", "Columns", profile["columns"]),
    ("⚠️", "Missing Values", profile["missing_total"]),
    ("♻️", "Duplicate Rows", profile["duplicate_rows"]),
]
for col, (icon, label, value) in zip((c1, c2, c3, c4), metrics):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{icon} {value:,}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# KEY INSIGHTS DASHBOARD — Budget, location, type stats
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">📈 Key Statistics</div>', unsafe_allow_html=True)
stat_cols = st.columns(4)
stat_idx = 0

if schema.primary_budget_col and schema.primary_budget_col in df.columns:
    series = pd.to_numeric(df[schema.primary_budget_col], errors="coerce").dropna()
    if not series.empty:
        stat_cols[0].markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">₹{series.mean():,.0f}</div>'
            f'<div class="metric-label">Avg {schema.primary_budget_col}</div></div>',
            unsafe_allow_html=True,
        )
        stat_cols[1].markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">₹{series.max():,.0f}</div>'
            f'<div class="metric-label">Highest {schema.primary_budget_col}</div></div>',
            unsafe_allow_html=True,
        )
        stat_idx = 2

if schema.location_col and schema.location_col in df.columns:
    top_loc = df[schema.location_col].value_counts().idxmax()
    stat_cols[stat_idx].markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">📍 {top_loc}</div>'
        f'<div class="metric-label">Top {schema.location_col}</div></div>',
        unsafe_allow_html=True,
    )
    stat_idx += 1

if schema.property_type_col and schema.property_type_col in df.columns and stat_idx < 4:
    top_type = df[schema.property_type_col].value_counts().idxmax()
    stat_cols[stat_idx].markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">🏠 {top_type}</div>'
        f'<div class="metric-label">Top {schema.property_type_col}</div></div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# DATA PREVIEW & SCHEMA
# --------------------------------------------------------------------------
with st.expander("🔍 Preview data & column details", expanded=False):
    st.dataframe(df.head(DATA_PREVIEW_ROWS), use_container_width=True)
    detail_rows = [
        {
            "Column": p.name,
            "Type": p.dtype,
            "Missing": p.missing,
            "Missing %": p.missing_pct,
            "Unique values": p.unique_count,
            "Role": p.role,
            "Sample values": ", ".join(map(str, p.sample_values)),
        }
        for p in profile["column_profiles"]
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

with st.expander("🧠 Detected schema (used by the query engine)", expanded=False):
    st.json(schema.to_dict())

# --------------------------------------------------------------------------
# AI / rule-based insights
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">✨ AI Insights</div>', unsafe_allow_html=True)
stat_lines = cached_rule_insights(df_h, df, schema)
insight_key = "|".join(stat_lines)
insights = stat_lines
if gemini_helper.is_configured():
    with st.spinner("Generating insights..."):
        insights = cached_gemini_insights(insight_key, stat_lines)
for line in insights:
    st.markdown(
        f'<div class="insight-pill">💡 {line}</div>',
        unsafe_allow_html=True,
    )

# Overview charts
with st.expander("📈 Dataset visual profile", expanded=False):
    overview_figs = charts.profile_overview_charts(df, schema)
    if not overview_figs:
        st.write("No suitable columns detected for automatic charts.")
    else:
        cols = st.columns(2)
        for i, (name, fig) in enumerate(overview_figs.items()):
            with cols[i % 2]:
                st.plotly_chart(
                    fig, use_container_width=True,
                    key=f"overview_{name}", config=CHART_CONFIG,
                )

st.divider()

# --------------------------------------------------------------------------
# SUGGESTED QUERY CARDS — Beautiful clickable cards
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">💬 Ask a question about your data</div>', unsafe_allow_html=True)

suggested: list[tuple[str, str]] = []  # (icon, question)
if schema.primary_budget_col:
    suggested.append(("📈", f"What is the average {schema.primary_budget_col}?"))
    # Guard: quantile only if column exists and has data
    budget_series = pd.to_numeric(df[schema.primary_budget_col], errors="coerce").dropna()
    if not budget_series.empty:
        q75 = int(budget_series.quantile(0.75))
        suggested.append(("💰", f"Show customers with {schema.primary_budget_col} above {q75}"))
if schema.property_type_col:
    suggested.append(("🏠", f"How many customers want each {schema.property_type_col}?"))
if schema.location_col:
    budget_label = schema.primary_budget_col or "value"
    suggested.append(("📍", f"Which {schema.location_col} has the highest average {budget_label}?"))
if schema.status_col:
    suggested.append(("📊", f"Give me a breakdown of {schema.status_col}"))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("**Suggested questions:**")
sugg_cols = st.columns(min(len(suggested), 4) or 1)
clicked_question = None
for i, (icon, q) in enumerate(suggested[:4]):
    with sugg_cols[i % len(sugg_cols)]:
        if st.button(f"{icon} {q}", key=f"sugg_{i}", use_container_width=True):
            clicked_question = q

question = st.chat_input("e.g. How many customers have a budget above 1 crore?")
final_question = clicked_question or question

# --------------------------------------------------------------------------
# QUERY PROCESSING with follow-up context
# --------------------------------------------------------------------------
if final_question:
    with st.spinner("⏳ Analyzing your question..."):
        columns = list(df.columns)
        schema_dict = schema.to_dict()

        intent = None
        used_gemini_intent = False
        if gemini_helper.is_configured():
            intent = cached_intent(final_question, columns, schema_dict)
            used_gemini_intent = intent is not None

        if intent is None:
            intent = rule_based_intent(final_question, schema, df)

        # Follow-up context: merge previous conditions if appropriate
        prev_conditions = st.session_state.get("last_conditions", [])
        q_lower = final_question.lower()
        is_follow_up = any(
            kw in q_lower
            for kw in ["only", "those", "of these", "among them", "from those", "sort them"]
        )
        if is_follow_up and prev_conditions:
            intent = merge_follow_up_conditions(intent, prev_conditions)
            logger.info("Merged follow-up conditions from previous query")

        engine = QueryEngine(df, schema)
        result = engine.execute(intent)

        # Store conditions for follow-up
        if result.success:
            st.session_state["last_conditions"] = intent.get("conditions", [])

        summary_text = None
        used_gemini_summary = False
        if result.success and gemini_helper.is_configured():
            payload = build_result_payload(result)
            summary_text = cached_summary(final_question, result.operation, payload)
            used_gemini_summary = summary_text is not None

        if summary_text is None:
            summary_text = fallback_summary(final_question, result)

        fig = (
            charts.auto_visualize(
                result.operation, result.table_result, schema, result.columns_used
            )
            if result.success
            else None
        )

        st.session_state.chat_history.append(
            {
                "question": final_question,
                "summary": summary_text,
                "result": result,
                "fig": fig,
                "used_gemini_intent": used_gemini_intent,
                "used_gemini_summary": used_gemini_summary,
                "intent": intent,
                "timestamp": time.strftime("%H:%M:%S"),
                "execution_time_ms": result.execution_time_ms,
            }
        )
        _trim_chat_history()

# --------------------------------------------------------------------------
# RENDER CHAT HISTORY (most recent first)
# --------------------------------------------------------------------------
for turn_idx, turn in enumerate(reversed(st.session_state.chat_history)):
    result = turn["result"]

    # User bubble
    st.markdown(
        f'<div class="chat-bubble-user">🧑 {turn["question"]}</div>',
        unsafe_allow_html=True,
    )

    # AI bubble
    st.markdown(
        f'<div class="chat-bubble-ai">🤖 {turn["summary"]}</div>',
        unsafe_allow_html=True,
    )

    # Confidence badge
    st.markdown(
        '<span class="confidence-badge confidence-badge-high">'
        "🟢 High Confidence — Computed directly from uploaded dataset. "
        "No AI estimation used.</span>",
        unsafe_allow_html=True,
    )

    # Data table + downloads
    if (
        result.success
        and result.table_result is not None
        and not result.table_result.empty
    ):
        st.dataframe(
            result.table_result.head(TABLE_DISPLAY_ROWS), use_container_width=True
        )
        dl1, dl2 = st.columns(2)
        csv_bytes = result.table_result.to_csv(index=False).encode("utf-8")
        dl1.download_button(
            "⬇️ Download filtered data (CSV)",
            csv_bytes,
            file_name=f"result_{turn_idx}.csv",
            mime="text/csv",
            key=f"dl_data_{turn_idx}",
        )
        summary_bytes = turn["summary"].encode("utf-8")
        dl2.download_button(
            "⬇️ Download summary (TXT)",
            summary_bytes,
            file_name=f"summary_{turn_idx}.txt",
            mime="text/plain",
            key=f"dl_summary_{turn_idx}",
        )

    # Chart
    if turn["fig"] is not None:
        st.plotly_chart(
            turn["fig"],
            use_container_width=True,
            key=f"chart_{turn_idx}",
            config=CHART_CONFIG,
        )

    # Explainability Panel
    with st.expander("🔎 How was this answer calculated?"):
        st.markdown(f"**Operation:** `{result.operation}`")
        st.markdown(f"**Explanation:** {result.explanation or 'N/A'}")
        st.markdown(
            f"**Columns used:** "
            f"{', '.join(result.columns_used) if result.columns_used else 'N/A'}"
        )
        st.markdown(f"**Rows scanned:** {result.rows_scanned:,}")
        st.markdown(f"**Rows matched:** {result.rows_matched:,}")
        st.markdown(f"**Filters applied:** {result.filters_applied}")
        st.markdown(f"**Execution time:** {result.execution_time_ms:.1f} ms")
        st.markdown(f"**Timestamp:** {turn.get('timestamp', 'N/A')}")

        st.markdown("---")
        st.markdown("**📋 Execution Pipeline:**")
        pipeline_html = f"""
        <div class="pipeline-step">
            <div class="pipeline-icon">1</div>
            <div><b>User Question:</b> {turn["question"]}</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-icon">2</div>
            <div><b>Intent Detection:</b> {'Gemini AI' if turn["used_gemini_intent"] else 'Rule-based fallback'} → <code>{result.operation}</code></div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-icon">3</div>
            <div><b>Pandas Execution:</b> {result.explanation or 'N/A'}</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-icon">4</div>
            <div><b>Result:</b> {'Scalar: ' + str(result.scalar_result) if result.scalar_result is not None else str(result.rows_matched) + ' rows'}</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-icon">5</div>
            <div><b>Summary:</b> {'Gemini AI phrasing' if turn["used_gemini_summary"] else 'Template-based'}</div>
        </div>
        """
        st.markdown(pipeline_html, unsafe_allow_html=True)

        st.caption(
            "Gemini only mapped this question to the operation/column above "
            "and phrased the final sentence. The actual number was computed "
            "by pandas — never by AI."
        )

        st.markdown("**Parsed intent:**")
        st.json(turn["intent"])

    st.divider()

if not st.session_state.chat_history:
    st.caption(
        "Ask a question above, or click one of the suggested questions to get started."
    )

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">'
    "Built with ❤️ using Streamlit · Pandas · Plotly · Google Gemini | "
    "Every answer is computed by pandas — zero hallucination guaranteed"
    "</div>",
    unsafe_allow_html=True,
)
