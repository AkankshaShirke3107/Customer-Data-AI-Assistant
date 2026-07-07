"""
app.py
------
Customer Data AI Assistant - "Chat with your Excel data using Natural
Language."

Streamlit front-end that ties together:
  utils.py         -> loading + dynamic schema detection + profiling
  query_engine.py  -> deterministic pandas execution (no hallucination)
  gemini_helper.py -> Gemini for intent understanding + summarization only
  charts.py        -> automatic Plotly visualizations

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import charts
import gemini_helper
from query_engine import QueryEngine, rule_based_intent
from utils import detect_schema, load_dataframe, profile_dataset, rule_based_insights

load_dotenv()

st.set_page_config(
    page_title="Customer Data AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# THEME / CSS
# --------------------------------------------------------------------------
def inject_css(dark_mode: bool) -> None:
    if dark_mode:
        bg, panel, text, sub, accent, border = "#0F1116", "#1A1D27", "#F2F3F7", "#9AA1B5", "#7C8CFF", "#2A2E3D"
    else:
        bg, panel, text, sub, accent, border = "#F7F8FC", "#FFFFFF", "#1A1D27", "#666E82", "#6366F1", "#E7E9F3"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        section[data-testid="stSidebar"] {{ background-color: {panel}; border-right: 1px solid {border}; }}
        .metric-card {{
            background: {panel}; border: 1px solid {border}; border-radius: 14px;
            padding: 18px 20px; text-align: left; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .metric-value {{ font-size: 28px; font-weight: 700; color: {accent}; }}
        .metric-label {{ font-size: 13px; color: {sub}; margin-top: 2px; }}
        .app-header {{
            padding: 22px 26px; border-radius: 18px; margin-bottom: 18px;
            background: linear-gradient(135deg, {accent}22, {accent}05);
            border: 1px solid {border};
        }}
        .app-title {{ font-size: 30px; font-weight: 800; color: {text}; margin-bottom: 2px;}}
        .app-tagline {{ font-size: 15px; color: {sub}; }}
        .insight-pill {{
            display:block; background:{panel}; border:1px solid {border}; border-radius:10px;
            padding:10px 14px; margin-bottom:8px; font-size:14px; color:{text};
        }}
        .chat-bubble-user {{
            background:{accent}; color:white; padding:10px 16px; border-radius:16px 16px 4px 16px;
            display:inline-block; margin:6px 0; max-width:80%;
        }}
        .chat-bubble-ai {{
            background:{panel}; border:1px solid {border}; color:{text}; padding:10px 16px;
            border-radius:16px 16px 16px 4px; display:inline-block; margin:6px 0; max-width:80%;
        }}
        .confidence-badge {{
            display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px;
            font-weight:600; background:{accent}22; color:{accent};
        }}
        div.stButton > button {{
            border-radius: 10px; border: 1px solid {border}; font-weight: 600;
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
def cached_profile(df: pd.DataFrame) -> dict:
    return profile_dataset(df)


@st.cache_data(show_spinner=False)
def cached_schema(df: pd.DataFrame):
    return detect_schema(df)


@st.cache_data(show_spinner=False)
def cached_rule_insights(df: pd.DataFrame, _schema) -> list[str]:
    return rule_based_insights(df, _schema)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_gemini_insights(stats_key: str, stats_lines: list[str]) -> list[str]:
    return gemini_helper.generate_insights(stats_lines)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_intent(question: str, columns: list[str], schema_dict: dict) -> dict | None:
    try:
        return gemini_helper.understand_intent(question, columns, schema_dict)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def cached_summary(question: str, operation: str, payload: dict) -> str | None:
    try:
        return gemini_helper.summarize_result(question, operation, payload)
    except Exception:
        return None


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
    if result.scalar_result is not None and (result.table_result is None or result.operation in {"count", "sum", "average", "min", "max", "distinct_count"}):
        val = result.scalar_result
        if isinstance(val, float):
            val = round(val, 2)
        return f"Result for **{result.operation}**: **{val}**"
    if result.table_result is not None:
        return f"Found **{len(result.table_result)}** matching row(s)."
    return "No result could be computed for this question."


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark mode", value=True)
    st.divider()

    st.markdown("## 📁 Data Source")
    uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx", "xls"])
    use_sample = False
    if uploaded_file is None:
        use_sample = st.checkbox("Use bundled sample dataset (Pune real-estate leads)", value=True)

    st.divider()
    st.markdown("## 🔑 Gemini API")
    if gemini_helper.is_configured():
        st.success("Gemini API key detected.")
    else:
        st.warning("No GEMINI_API_KEY found. The app will still work using a rule-based fallback engine, but AI summaries/insights will be simplified.")
        manual_key = st.text_input("Paste a Gemini API key for this session (optional)", type="password")
        if manual_key:
            os.environ["GEMINI_API_KEY"] = manual_key
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.pop("chat_history", None)
        st.rerun()

inject_css(dark_mode)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">📊 Customer Data AI Assistant</div>
        <div class="app-tagline">Chat with your Excel data using Natural Language.</div>
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
        df = cached_load_dataframe(uploaded_file.getvalue())
    elif use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_leads.xlsx")
        with open(sample_path, "rb") as f:
            df = cached_load_dataframe(f.read())
except ValueError as exc:
    load_error = str(exc)

if load_error:
    st.error(f"❌ {load_error}")
    st.stop()

if df is None:
    st.info("👋 Upload an Excel file from the sidebar (or check *'Use bundled sample dataset'*) to get started.")
    st.stop()

schema = cached_schema(df)
profile = cached_profile(df)

# --------------------------------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------------------------------
st.markdown("### 📋 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
for col, label, value in zip(
    (c1, c2, c3, c4),
    ("Rows", "Columns", "Missing Values", "Duplicate Rows"),
    (profile["rows"], profile["columns"], profile["missing_total"], profile["duplicate_rows"]),
):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

with st.expander("🔍 Preview data & column details", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    detail_rows = [
        {
            "Column": p.name, "Type": p.dtype, "Missing": p.missing,
            "Missing %": p.missing_pct, "Unique values": p.unique_count,
            "Role": p.role, "Sample values": ", ".join(map(str, p.sample_values)),
        }
        for p in profile["column_profiles"]
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

with st.expander("🧠 Detected schema (used by the query engine)", expanded=False):
    st.json(schema.to_dict())

# AI / rule-based insights
st.markdown("### ✨ AI Insights")
stat_lines = cached_rule_insights(df, schema)
insight_key = "|".join(stat_lines)
insights = stat_lines
if gemini_helper.is_configured():
    with st.spinner("Generating insights..."):
        insights = cached_gemini_insights(insight_key, stat_lines)
for line in insights:
    st.markdown(f'<div class="insight-pill">💡 {line}</div>', unsafe_allow_html=True)

# Overview charts
with st.expander("📈 Dataset visual profile", expanded=False):
    overview_figs = charts.profile_overview_charts(df, schema)
    if not overview_figs:
        st.write("No suitable columns detected for automatic charts.")
    else:
        cols = st.columns(2)
        for i, (name, fig) in enumerate(overview_figs.items()):
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True, key=f"overview_{name}")

st.divider()

# --------------------------------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------------------------------
st.markdown("### 💬 Ask a question about your data")

suggested = []
if schema.primary_budget_col:
    suggested.append(f"What is the average {schema.primary_budget_col}?")
    suggested.append(f"Show customers with {schema.primary_budget_col} above {int(df[schema.primary_budget_col].quantile(0.75))}")
if schema.property_type_col:
    suggested.append(f"How many customers want each {schema.property_type_col}?")
if schema.location_col:
    suggested.append(f"Which {schema.location_col} has the highest average {schema.primary_budget_col or 'value'}?")
if schema.status_col:
    suggested.append(f"Give me a breakdown of {schema.status_col}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("**Suggested questions:**")
sugg_cols = st.columns(min(len(suggested), 4) or 1)
clicked_question = None
for i, q in enumerate(suggested[:4]):
    if sugg_cols[i % len(sugg_cols)].button(q, key=f"sugg_{i}"):
        clicked_question = q

question = st.chat_input("e.g. How many customers have a budget above 1 crore?")
final_question = clicked_question or question

if final_question:
    with st.spinner("Thinking..."):
        columns = list(df.columns)
        schema_dict = schema.to_dict()

        intent = None
        used_gemini_intent = False
        if gemini_helper.is_configured():
            intent = cached_intent(final_question, columns, schema_dict)
            used_gemini_intent = intent is not None

        if intent is None:
            intent = rule_based_intent(final_question, schema, df)

        engine = QueryEngine(df, schema)
        result = engine.execute(intent)

        summary_text = None
        used_gemini_summary = False
        if result.success and gemini_helper.is_configured():
            payload = build_result_payload(result)
            summary_text = cached_summary(final_question, result.operation, payload)
            used_gemini_summary = summary_text is not None

        if summary_text is None:
            summary_text = fallback_summary(final_question, result)

        fig = charts.auto_visualize(result.operation, result.table_result, schema, result.columns_used) if result.success else None

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
            }
        )

# Render chat history (most recent first)
for turn_idx, turn in enumerate(reversed(st.session_state.chat_history)):
    result = turn["result"]
    st.markdown(f'<div class="chat-bubble-user">🧑 {turn["question"]}</div>', unsafe_allow_html=True)

    confidence = "High (Gemini + Pandas)" if turn["used_gemini_summary"] else "Medium (rule-based fallback)"
    badge_color = "🟢" if turn["used_gemini_summary"] else "🟡"
    st.markdown(
        f'<div class="chat-bubble-ai">🤖 {turn["summary"]}</div> '
        f'<span class="confidence-badge">{badge_color} Confidence: {confidence}</span>',
        unsafe_allow_html=True,
    )

    if result.success and result.table_result is not None and not result.table_result.empty:
        st.dataframe(result.table_result.head(200), use_container_width=True)
        dl1, dl2 = st.columns(2)
        csv_bytes = result.table_result.to_csv(index=False).encode("utf-8")
        dl1.download_button(
            "⬇️ Download filtered data (CSV)", csv_bytes,
            file_name=f"result_{turn_idx}.csv", mime="text/csv", key=f"dl_data_{turn_idx}",
        )
        summary_bytes = turn["summary"].encode("utf-8")
        dl2.download_button(
            "⬇️ Download summary (TXT)", summary_bytes,
            file_name=f"summary_{turn_idx}.txt", mime="text/plain", key=f"dl_summary_{turn_idx}",
        )

    if turn["fig"] is not None:
        st.plotly_chart(turn["fig"], use_container_width=True, key=f"chart_{turn_idx}")

    with st.expander("🔎 How was this answer calculated?"):
        st.write(f"**Operation:** `{result.operation}`")
        st.write(f"**Explanation:** {result.explanation or 'N/A'}")
        st.write(f"**Columns used:** {', '.join(result.columns_used) if result.columns_used else 'N/A'}")
        st.write(f"**Parsed intent (from Gemini or fallback parser):**")
        st.json(turn["intent"])
        st.caption(
            "Gemini only mapped this question to the operation/column above and phrased "
            "the final sentence. The actual number was computed by pandas."
        )

    st.divider()

if not st.session_state.chat_history:
    st.caption("Ask a question above, or click one of the suggested questions to get started.")
