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


# ==========================================================================
# PREMIUM CSS — Vercel / Linear / Stripe-inspired dark theme
# ==========================================================================
def inject_premium_css() -> None:
    st.markdown(
        """
        <style>
        /* ================================================================
           FONTS
           ================================================================ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg:          #09090B;
            --surface:     #111113;
            --card:        #18181B;
            --card-hover:  #1F1F23;
            --border:      rgba(255,255,255,.06);
            --border-sub:  rgba(255,255,255,.04);
            --text:        #FAFAFA;
            --text-2:      #A1A1AA;
            --text-3:      #71717A;
            --primary:     #3B82F6;
            --primary-dim: rgba(59,130,246,.12);
            --secondary:   #8B5CF6;
            --success:     #22C55E;
            --success-dim: rgba(34,197,94,.10);
            --warning:     #F59E0B;
            --danger:      #EF4444;
            --radius:      14px;
            --radius-lg:   20px;
            --radius-sm:   8px;
            --font:        'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --mono:        'JetBrains Mono', 'Fira Code', monospace;
            --shadow:      0 0 0 1px var(--border), 0 2px 12px rgba(0,0,0,.4);
            --shadow-lg:   0 0 0 1px var(--border), 0 8px 40px rgba(0,0,0,.5);
            --transition:  all .2s cubic-bezier(.4,0,.2,1);
        }

        /* ================================================================
           GLOBAL RESETS
           ================================================================ */
        *, html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stMarkdown, .stText, p, span, label, li, td, th, h1, h2, h3, h4, h5, h6,
        input, textarea, button, select, code, pre,
        div[data-testid="stChatMessage"] p {
            font-family: var(--font) !important;
        }
        code, pre, .stCode {
            font-family: var(--mono) !important;
        }

        .stApp {
            background: var(--bg) !important;
            color: var(--text) !important;
        }

        /* Hide Streamlit branding */
        #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden !important; height: 0 !important; }
        .stDeployButton { display: none !important; }
        div[data-testid="stDecoration"] { display: none !important; }
        div[data-testid="stStatusWidget"] { display: none !important; }

        /* Main container spacing */
        .block-container {
            padding: 2rem 3rem 4rem 3rem !important;
            max-width: 1280px !important;
        }

        /* ================================================================
           SIDEBAR — Navigation Panel
           ================================================================ */
        section[data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding: 1.5rem 1.25rem !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-2) !important;
            font-size: 13px !important;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stFileUploader label {
            color: var(--text-2) !important;
            font-size: 12px !important;
            text-transform: uppercase !important;
            letter-spacing: .06em !important;
            font-weight: 600 !important;
        }

        /* Sidebar section labels */
        .sidebar-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--text-3);
            margin: 20px 0 8px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border-sub);
        }
        .sidebar-version {
            font-size: 11px;
            color: var(--text-3);
            text-align: center;
            padding: 16px 0 4px;
            border-top: 1px solid var(--border-sub);
            margin-top: 24px;
        }

        /* ================================================================
           HERO SECTION
           ================================================================ */
        .hero {
            position: relative;
            padding: 48px 0 40px;
            margin-bottom: 8px;
        }
        .hero::after {
            content: '';
            position: absolute;
            top: 0; left: 50%;
            transform: translateX(-50%);
            width: 600px; height: 300px;
            background: radial-gradient(ellipse, rgba(59,130,246,.06) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }
        .hero-content {
            position: relative;
            z-index: 1;
            text-align: center;
        }
        .hero-badge-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 20px;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: .02em;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--text-2);
        }
        .hero-badge .dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            display: inline-block;
        }
        .hero-badge .dot-green { background: var(--success); box-shadow: 0 0 6px var(--success); }
        .hero-badge .dot-blue  { background: var(--primary); box-shadow: 0 0 6px var(--primary); }
        .hero-badge .dot-purple { background: var(--secondary); box-shadow: 0 0 6px var(--secondary); }

        .hero-title {
            font-size: 44px;
            font-weight: 800;
            letter-spacing: -1.5px;
            line-height: 1.1;
            color: var(--text);
            margin: 0 0 12px;
            background: linear-gradient(135deg, #FAFAFA 0%, #A1A1AA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            font-size: 16px;
            font-weight: 400;
            color: var(--text-3);
            max-width: 540px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ================================================================
           KPI / METRIC CARDS — Glass Bento Grid
           ================================================================ */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 8px 0 28px;
        }
        @media (max-width: 768px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .kpi-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px 18px 16px;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--primary), transparent);
            opacity: 0;
            transition: opacity .3s ease;
        }
        .kpi-card:hover {
            background: var(--card-hover);
            border-color: rgba(255,255,255,.10);
            transform: translateY(-2px);
        }
        .kpi-card:hover::before { opacity: 1; }
        .kpi-icon {
            width: 32px; height: 32px;
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
            margin-bottom: 12px;
        }
        .kpi-icon-blue    { background: var(--primary-dim); color: var(--primary); }
        .kpi-icon-purple  { background: rgba(139,92,246,.12); color: var(--secondary); }
        .kpi-icon-amber   { background: rgba(245,158,11,.10); color: var(--warning); }
        .kpi-icon-red     { background: rgba(239,68,68,.10); color: var(--danger); }
        .kpi-icon-green   { background: var(--success-dim); color: var(--success); }
        .kpi-value {
            font-size: 26px;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -.5px;
            line-height: 1.2;
        }
        .kpi-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-3);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        /* ================================================================
           SECTION HEADERS
           ================================================================ */
        .section-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: var(--text-3);
            margin: 36px 0 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-title::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        /* ================================================================
           INSIGHT CARDS
           ================================================================ */
        .insight-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 14px 18px;
            margin-bottom: 8px;
            font-size: 13.5px;
            color: var(--text-2);
            line-height: 1.55;
            transition: var(--transition);
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        .insight-card:hover {
            background: var(--card-hover);
            border-color: rgba(255,255,255,.10);
        }
        .insight-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--primary);
            flex-shrink: 0;
            margin-top: 7px;
        }

        /* ================================================================
           SUGGESTED QUESTION CHIPS
           ================================================================ */
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 12px 0 20px;
        }

        /* ================================================================
           CHAT EXPERIENCE — ChatGPT-style
           ================================================================ */
        .chat-turn {
            padding: 24px 0;
            border-bottom: 1px solid var(--border-sub);
            animation: fadeInUp .35s ease;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .chat-row {
            display: flex;
            gap: 14px;
            align-items: flex-start;
            max-width: 860px;
        }
        .chat-row-user { justify-content: flex-end; margin-left: auto; }
        .chat-avatar {
            width: 30px; height: 30px;
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            flex-shrink: 0;
        }
        .avatar-user { background: var(--primary-dim); color: var(--primary); }
        .avatar-ai   { background: rgba(139,92,246,.12); color: var(--secondary); }

        .chat-msg-user {
            background: var(--primary);
            color: #fff;
            padding: 10px 16px;
            border-radius: 16px 16px 4px 16px;
            font-size: 14px;
            font-weight: 500;
            max-width: 70%;
            line-height: 1.5;
        }
        .chat-msg-ai {
            background: var(--card);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 14px 18px;
            border-radius: 4px 16px 16px 16px;
            font-size: 14px;
            line-height: 1.6;
            max-width: 85%;
        }
        .chat-meta {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 8px;
            padding-left: 44px;
        }
        .chat-timestamp {
            font-size: 11px;
            color: var(--text-3);
            font-family: var(--mono) !important;
        }
        .chat-exec-time {
            font-size: 11px;
            color: var(--text-3);
            font-family: var(--mono) !important;
        }

        /* Confidence badge */
        .conf-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            background: var(--success-dim);
            color: var(--success);
            border: 1px solid rgba(34,197,94,.15);
        }
        .conf-dot {
            width: 5px; height: 5px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 4px var(--success);
        }

        /* ================================================================
           EXPLAINABILITY PIPELINE — Timeline
           ================================================================ */
        .pipeline {
            position: relative;
            padding-left: 28px;
            margin: 12px 0;
        }
        .pipeline::before {
            content: '';
            position: absolute;
            left: 11px;
            top: 8px;
            bottom: 8px;
            width: 1px;
            background: var(--border);
        }
        .pipe-step {
            position: relative;
            padding: 8px 0 8px 16px;
            font-size: 13px;
            color: var(--text-2);
            line-height: 1.5;
        }
        .pipe-step::before {
            content: '';
            position: absolute;
            left: -20px;
            top: 14px;
            width: 7px; height: 7px;
            border-radius: 50%;
            background: var(--primary);
            border: 2px solid var(--bg);
            box-shadow: 0 0 0 2px var(--primary-dim);
        }
        .pipe-step:last-child::before {
            background: var(--success);
            box-shadow: 0 0 0 2px var(--success-dim);
        }
        .pipe-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: var(--text-3);
            margin-bottom: 2px;
        }
        .pipe-value {
            color: var(--text);
            font-weight: 500;
        }
        .pipe-value code {
            background: var(--surface);
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 12px;
            color: var(--primary);
            border: 1px solid var(--border);
        }

        /* Stats row in explainability */
        .stat-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin: 12px 0;
        }
        .stat-mini {
            background: var(--surface);
            border: 1px solid var(--border-sub);
            border-radius: var(--radius-sm);
            padding: 10px 12px;
            text-align: center;
        }
        .stat-mini-val {
            font-size: 16px;
            font-weight: 700;
            color: var(--text);
            font-family: var(--mono) !important;
        }
        .stat-mini-lbl {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: var(--text-3);
            margin-top: 2px;
        }

        /* ================================================================
           EMPTY STATE
           ================================================================ */
        .empty-state {
            text-align: center;
            padding: 80px 20px;
        }
        .empty-icon {
            width: 64px; height: 64px;
            border-radius: 16px;
            background: var(--card);
            border: 1px solid var(--border);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            margin-bottom: 20px;
        }
        .empty-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 8px;
        }
        .empty-desc {
            font-size: 14px;
            color: var(--text-3);
            max-width: 400px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ================================================================
           CHART CONTAINER
           ================================================================ */
        .chart-wrap {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px;
            margin: 12px 0;
        }

        /* ================================================================
           BUTTONS
           ================================================================ */
        div.stButton > button {
            background: var(--card) !important;
            color: var(--text-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            padding: 6px 14px !important;
            transition: var(--transition) !important;
        }
        div.stButton > button:hover {
            background: var(--card-hover) !important;
            color: var(--text) !important;
            border-color: rgba(255,255,255,.12) !important;
        }
        div.stButton > button:active {
            transform: scale(.98) !important;
        }

        /* Download buttons */
        div.stDownloadButton > button {
            background: var(--surface) !important;
            color: var(--text-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            transition: var(--transition) !important;
        }
        div.stDownloadButton > button:hover {
            background: var(--card) !important;
            color: var(--text) !important;
        }

        /* ================================================================
           EXPANDER
           ================================================================ */
        .streamlit-expanderHeader {
            font-size: 13px !important;
            font-weight: 600 !important;
            color: var(--text-2) !important;
            background: var(--surface) !important;
            border-radius: var(--radius-sm) !important;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            background: var(--card) !important;
        }

        /* ================================================================
           DATAFRAME
           ================================================================ */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
        }

        /* ================================================================
           CHAT INPUT
           ================================================================ */
        [data-testid="stChatInput"] {
            border-color: var(--border) !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px var(--primary-dim) !important;
        }

        /* ================================================================
           FILE UPLOADER
           ================================================================ */
        [data-testid="stFileUploader"] {
            border: 1px dashed var(--border) !important;
            border-radius: var(--radius) !important;
            transition: var(--transition) !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(255,255,255,.15) !important;
        }

        /* ================================================================
           DIVIDER
           ================================================================ */
        hr {
            border-color: var(--border-sub) !important;
            margin: 8px 0 !important;
        }

        /* ================================================================
           FOOTER
           ================================================================ */
        .app-footer {
            text-align: center;
            padding: 32px 0 8px;
            font-size: 12px;
            color: var(--text-3);
            border-top: 1px solid var(--border-sub);
            margin-top: 48px;
        }
        .app-footer a {
            color: var(--text-2);
            text-decoration: none;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.08); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,.15); }

        /* ================================================================
           ANIMATIONS
           ================================================================ */
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        .fade-in { animation: fadeIn .4s ease; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# CACHED HELPERS (unchanged backend logic)
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


# ==========================================================================
# SIDEBAR — Premium Navigation Panel
# ==========================================================================
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:12px 0 4px;">'
        '<span style="font-size:18px;font-weight:700;letter-spacing:-.5px;'
        'color:#FAFAFA;">DataLens</span>'
        '<span style="font-size:10px;color:#71717A;margin-left:6px;'
        'font-weight:500;">AI</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Data Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Excel (.xlsx)", type=["xlsx", "xls"], label_visibility="collapsed"
    )
    use_sample = False
    if uploaded_file is None:
        use_sample = st.checkbox("Use sample dataset", value=True)

    st.markdown('<div class="sidebar-label">AI Configuration</div>', unsafe_allow_html=True)
    if gemini_helper.is_configured():
        st.markdown(
            '<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#22C55E;">'
            '<span style="width:6px;height:6px;border-radius:50%;background:#22C55E;'
            'box-shadow:0 0 6px #22C55E;display:inline-block;"></span>'
            'Gemini API connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:12px;color:#F59E0B;">No API key — using fallback engine</div>',
            unsafe_allow_html=True,
        )
        manual_key = st.text_input(
            "API Key", type="password", label_visibility="collapsed",
            placeholder="Paste Gemini API key..."
        )
        if manual_key:
            st.session_state["_gemini_key"] = manual_key
            os.environ["GEMINI_API_KEY"] = manual_key
            st.rerun()

    st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("last_conditions", None)
        st.rerun()

    st.markdown(
        '<div class="sidebar-version">'
        'Customer Data AI Assistant · v2.0<br>'
        '<span style="font-size:10px;">Pandas engine · Zero hallucination</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# Inject CSS
inject_premium_css()


# ==========================================================================
# HERO SECTION
# ==========================================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <div class="hero-badge-row">
                <span class="hero-badge"><span class="dot dot-green"></span>Zero Hallucination</span>
                <span class="hero-badge"><span class="dot dot-blue"></span>Powered by Pandas</span>
                <span class="hero-badge"><span class="dot dot-purple"></span>AI Assisted</span>
            </div>
            <h1 class="hero-title">Customer Data AI Assistant</h1>
            <p class="hero-subtitle">
                Ask questions about your Excel data in plain English.
                Every answer is computed deterministically by Pandas — never guessed by AI.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# DATA LOADING (backend unchanged)
# ==========================================================================
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
    st.markdown(
        """
        <div class="empty-state fade-in">
            <div class="empty-icon">↑</div>
            <div class="empty-title">No dataset loaded</div>
            <div class="empty-desc">
                Upload an Excel file from the sidebar, or enable the sample
                dataset to start exploring instantly.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

df_h = _df_hash(df)
schema = cached_schema(df_h, df)
profile = cached_profile(df_h, df)


# ==========================================================================
# KPI CARDS — Bento Grid
# ==========================================================================
st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

kpi_html = '<div class="kpi-grid">'
kpi_items = [
    ("kpi-icon-blue",   "⊞", "Rows",           f"{profile['rows']:,}"),
    ("kpi-icon-purple", "⊟", "Columns",         f"{profile['columns']}"),
    ("kpi-icon-amber",  "⚠", "Missing Values",  f"{profile['missing_total']:,}"),
    ("kpi-icon-red",    "⊗", "Duplicates",       f"{profile['duplicate_rows']:,}"),
]
for icon_cls, icon_char, label, value in kpi_items:
    kpi_html += f"""
    <div class="kpi-card">
        <div class="kpi-icon {icon_cls}">{icon_char}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """
kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)


# ==========================================================================
# KEY STATISTICS — Second row of KPI cards
# ==========================================================================
stats_html = '<div class="kpi-grid">'
stat_count = 0

if schema.primary_budget_col and schema.primary_budget_col in df.columns:
    series = pd.to_numeric(df[schema.primary_budget_col], errors="coerce").dropna()
    if not series.empty:
        stats_html += f"""
        <div class="kpi-card">
            <div class="kpi-icon kpi-icon-green">₹</div>
            <div class="kpi-value">₹{series.mean():,.0f}</div>
            <div class="kpi-label">Avg {schema.primary_budget_col}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon kpi-icon-blue">↑</div>
            <div class="kpi-value">₹{series.max():,.0f}</div>
            <div class="kpi-label">Highest {schema.primary_budget_col}</div>
        </div>
        """
        stat_count = 2

if schema.location_col and schema.location_col in df.columns:
    top_loc = df[schema.location_col].value_counts().idxmax()
    stats_html += f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-purple">◎</div>
        <div class="kpi-value">{top_loc}</div>
        <div class="kpi-label">Top {schema.location_col}</div>
    </div>
    """
    stat_count += 1

if schema.property_type_col and schema.property_type_col in df.columns and stat_count < 4:
    top_type = df[schema.property_type_col].value_counts().idxmax()
    stats_html += f"""
    <div class="kpi-card">
        <div class="kpi-icon kpi-icon-amber">◈</div>
        <div class="kpi-value">{top_type}</div>
        <div class="kpi-label">Top {schema.property_type_col}</div>
    </div>
    """

stats_html += '</div>'
if stat_count > 0:
    st.markdown(stats_html, unsafe_allow_html=True)


# ==========================================================================
# DATA PREVIEW & SCHEMA
# ==========================================================================
with st.expander("Data Preview & Schema", expanded=False):
    st.dataframe(df.head(DATA_PREVIEW_ROWS), use_container_width=True)
    detail_rows = [
        {
            "Column": p.name,
            "Type": p.dtype,
            "Missing": p.missing,
            "Missing %": p.missing_pct,
            "Unique": p.unique_count,
            "Role": p.role,
            "Samples": ", ".join(map(str, p.sample_values)),
        }
        for p in profile["column_profiles"]
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

with st.expander("Detected Schema", expanded=False):
    st.json(schema.to_dict())


# ==========================================================================
# AI INSIGHTS
# ==========================================================================
st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)
stat_lines = cached_rule_insights(df_h, df, schema)
insight_key = "|".join(stat_lines)
insights = stat_lines
if gemini_helper.is_configured():
    with st.spinner("Analyzing dataset..."):
        insights = cached_gemini_insights(insight_key, stat_lines)

for line in insights:
    st.markdown(
        f'<div class="insight-card">'
        f'<span class="insight-dot"></span>'
        f'<span>{line}</span></div>',
        unsafe_allow_html=True,
    )


# Overview charts
with st.expander("Visual Profile", expanded=False):
    overview_figs = charts.profile_overview_charts(df, schema)
    if not overview_figs:
        st.write("No suitable columns detected for automatic charts.")
    else:
        cols = st.columns(2)
        for i, (name, fig) in enumerate(overview_figs.items()):
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#A1A1AA", family="Inter, sans-serif"),
            )
            with cols[i % 2]:
                st.plotly_chart(
                    fig, use_container_width=True,
                    key=f"overview_{name}", config=CHART_CONFIG,
                )


# ==========================================================================
# SUGGESTED QUESTIONS — Chip Row
# ==========================================================================
st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)

suggested: list[str] = []
if schema.primary_budget_col:
    suggested.append(f"What is the average {schema.primary_budget_col}?")
    budget_series = pd.to_numeric(df[schema.primary_budget_col], errors="coerce").dropna()
    if not budget_series.empty:
        q75 = int(budget_series.quantile(0.75))
        suggested.append(f"Show customers with {schema.primary_budget_col} above {q75}")
if schema.property_type_col:
    suggested.append(f"How many customers want each {schema.property_type_col}?")
if schema.location_col:
    budget_label = schema.primary_budget_col or "value"
    suggested.append(f"Which {schema.location_col} has the highest average {budget_label}?")
if schema.status_col:
    suggested.append(f"Breakdown of {schema.status_col}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

sugg_cols = st.columns(min(len(suggested), 4) or 1)
clicked_question = None
for i, q in enumerate(suggested[:4]):
    with sugg_cols[i % len(sugg_cols)]:
        if st.button(q, key=f"sugg_{i}", use_container_width=True):
            clicked_question = q

question = st.chat_input("Ask anything about your data...")
final_question = clicked_question or question


# ==========================================================================
# QUERY PROCESSING (backend unchanged)
# ==========================================================================
if final_question:
    with st.spinner("Processing..."):
        columns = list(df.columns)
        schema_dict = schema.to_dict()

        intent = None
        used_gemini_intent = False
        if gemini_helper.is_configured():
            intent = cached_intent(final_question, columns, schema_dict)
            used_gemini_intent = intent is not None

        if intent is None:
            intent = rule_based_intent(final_question, schema, df)

        # Follow-up context
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


# ==========================================================================
# RENDER CHAT HISTORY — ChatGPT-style conversation
# ==========================================================================
for turn_idx, turn in enumerate(reversed(st.session_state.chat_history)):
    result = turn["result"]

    st.markdown('<div class="chat-turn">', unsafe_allow_html=True)

    # User message
    st.markdown(
        f'<div class="chat-row chat-row-user">'
        f'<div class="chat-msg-user">{turn["question"]}</div>'
        f'<div class="chat-avatar avatar-user">U</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # AI message
    st.markdown(
        f'<div class="chat-row" style="margin-top:12px;">'
        f'<div class="chat-avatar avatar-ai">AI</div>'
        f'<div class="chat-msg-ai">{turn["summary"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Meta row: confidence, timestamp, exec time
    st.markdown(
        f'<div class="chat-meta">'
        f'<span class="conf-badge"><span class="conf-dot"></span>Deterministic</span>'
        f'<span class="chat-timestamp">{turn.get("timestamp", "")}</span>'
        f'<span class="chat-exec-time">{turn["execution_time_ms"]:.0f}ms</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

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
            "Download CSV",
            csv_bytes,
            file_name=f"result_{turn_idx}.csv",
            mime="text/csv",
            key=f"dl_data_{turn_idx}",
        )
        summary_bytes = turn["summary"].encode("utf-8")
        dl2.download_button(
            "Download Summary",
            summary_bytes,
            file_name=f"summary_{turn_idx}.txt",
            mime="text/plain",
            key=f"dl_summary_{turn_idx}",
        )

    # Chart
    if turn["fig"] is not None:
        fig = turn["fig"]
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A1A1AA", family="Inter, sans-serif"),
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"chart_{turn_idx}",
            config=CHART_CONFIG,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Explainability Panel — Timeline
    with st.expander("Execution Details"):
        # Stats grid
        st.markdown(
            f"""
            <div class="stat-row">
                <div class="stat-mini">
                    <div class="stat-mini-val">{result.rows_scanned:,}</div>
                    <div class="stat-mini-lbl">Rows Scanned</div>
                </div>
                <div class="stat-mini">
                    <div class="stat-mini-val">{result.rows_matched:,}</div>
                    <div class="stat-mini-lbl">Rows Matched</div>
                </div>
                <div class="stat-mini">
                    <div class="stat-mini-val">{result.filters_applied}</div>
                    <div class="stat-mini-lbl">Filters</div>
                </div>
                <div class="stat-mini">
                    <div class="stat-mini-val">{result.execution_time_ms:.1f}ms</div>
                    <div class="stat-mini-lbl">Execution</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Pipeline timeline
        intent_method = "Gemini AI" if turn["used_gemini_intent"] else "Rule-based"
        summary_method = "Gemini AI" if turn["used_gemini_summary"] else "Template"
        scalar_display = (
            f"Scalar: {result.scalar_result}"
            if result.scalar_result is not None
            else f"{result.rows_matched} rows"
        )
        cols_display = ", ".join(result.columns_used) if result.columns_used else "—"

        st.markdown(
            f"""
            <div class="pipeline">
                <div class="pipe-step">
                    <div class="pipe-label">User Question</div>
                    <div class="pipe-value">{turn["question"]}</div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-label">Intent Detection · {intent_method}</div>
                    <div class="pipe-value"><code>{result.operation}</code> on [{cols_display}]</div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-label">Pandas Execution</div>
                    <div class="pipe-value">{result.explanation or "—"}</div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-label">Result</div>
                    <div class="pipe-value">{scalar_display}</div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-label">Summary · {summary_method}</div>
                    <div class="pipe-value">Natural language phrasing of computed result</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Every number was computed by Pandas. Gemini only classified the "
            "intent and phrased the result — it never computed a value."
        )

        with st.expander("Raw Intent JSON"):
            st.json(turn["intent"])


# Empty state hint
if not st.session_state.chat_history:
    st.markdown(
        '<div style="text-align:center;padding:40px 0;color:#71717A;font-size:13px;">'
        'Type a question above or click a suggestion to get started.'
        '</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# FOOTER
# ==========================================================================
st.markdown(
    '<div class="app-footer">'
    'Customer Data AI Assistant · Built with Streamlit, Pandas, Plotly & Google Gemini<br>'
    '<span style="font-size:11px;">Every answer is deterministic. Zero hallucination by design.</span>'
    '</div>',
    unsafe_allow_html=True,
)
