"""
app.py
------
Customer Data AI Assistant — "Chat with your Excel data using Natural Language."
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
from charts import CHART_CONFIG
from config import CACHE_TTL_SECONDS, DATA_PREVIEW_ROWS, TABLE_DISPLAY_ROWS
from query_engine import QueryEngine, merge_follow_up_conditions, rule_based_intent
from utils import detect_schema, load_dataframe, profile_dataset, rule_based_insights, validate_upload

load_dotenv()

st.set_page_config(
    page_title="Customer Data AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject CSS
with open(os.path.join(os.path.dirname(__file__), "style.css"), "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# SVG Icons
ICONS = {
    "upload": '<svg class="icon" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>',
    "database": '<svg class="icon" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>',
    "rows": '<svg class="icon" viewBox="0 0 24 24"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>',
    "cols": '<svg class="icon" viewBox="0 0 24 24"><line x1="12" y1="3" x2="12" y2="21"></line><line x1="6" y1="3" x2="6" y2="21"></line><line x1="18" y1="3" x2="18" y2="21"></line></svg>',
    "alert": '<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
    "check": '<svg class="icon" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    "bot": '<svg class="icon" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>',
    "user": '<svg class="icon" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>',
    "sparkle": '<svg class="icon" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>',
    "cpu": '<svg class="icon" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>',
}

# --------------------------------------------------------------------------
# CACHED HELPERS (Backend Unchanged)
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
    return f"{len(df)}_{df.shape[1]}_{hash(tuple(df.columns))}_{hash(tuple(df.dtypes))}"

def build_result_payload(result) -> dict:
    payload = {"operation": result.operation, "scalar_result": result.scalar_result}
    if result.table_result is not None and not result.table_result.empty:
        payload["table_preview"] = result.table_result.head(15).to_dict(orient="records")
        payload["row_count"] = len(result.table_result)
    return payload

def fallback_summary(question: str, result) -> str:
    if not result.success:
        return f"Error: {result.error}"
    if result.scalar_result is not None and (
        result.table_result is None or result.operation in {"count", "sum", "average", "min", "max", "distinct_count"}
    ):
        val = result.scalar_result
        if isinstance(val, float): val = round(val, 2)
        return f"The computed {result.operation} is {val}."
    if result.table_result is not None:
        return f"I found {len(result.table_result)} rows matching your criteria."
    return "Query executed successfully."

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------------------------------
# STATE INITIALIZATION
# --------------------------------------------------------------------------
df = None
load_error = None

# Sidebar is now ONLY for system settings, hidden by default
with st.sidebar:
    st.markdown("### Settings")
    if gemini_helper.is_configured():
        st.success("Gemini API Connected")
    else:
        st.warning("Gemini API Missing")
        st.session_state["_gemini_key"] = st.text_input("API Key", type="password")
        if st.session_state["_gemini_key"]:
            os.environ["GEMINI_API_KEY"] = st.session_state["_gemini_key"]
            st.rerun()

# Welcome Mode Wrapper
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

placeholder = st.empty()

with placeholder.container():
    st.markdown(
        '''
        <div class="welcome-hero">
            <div style="display:inline-flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:99px;font-size:12px;font-weight:600;margin-bottom:24px;color:var(--text-2);">
                <span style="color:var(--primary);">✦</span> Customer Data AI Assistant
            </div>
            <h1 class="welcome-title">Chat with your Excel data using Natural Language.</h1>
            <p class="welcome-subtitle">
                Enterprise-grade analytics powered by Pandas and Gemini. 
                Upload your dataset to instantly generate insights, charts, and deterministic answers with zero hallucinations.
            </p>
        </div>
        ''', unsafe_allow_html=True
    )
    
    col_up1, col_up2, col_up3 = st.columns([1, 2, 1])
    with col_up2:
        uploaded = st.file_uploader("Drop your dataset here", type=["xlsx", "xls"], label_visibility="collapsed")
        if uploaded:
            st.session_state.uploaded_file = uploaded
            st.rerun()
        
        use_sample = st.button("Try Demo Dataset Instead", use_container_width=True)
        if use_sample:
            sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_leads.xlsx")
            with open(sample_path, "rb") as f:
                st.session_state.uploaded_file = io.BytesIO(f.read())
                st.session_state.uploaded_file.name = "sample_leads.xlsx"
            st.rerun()

    st.markdown(
        f'''
        <div class="welcome-features">
            <div class="welcome-feature">
                {ICONS["database"]}
                <h3>Deterministic Engine</h3>
                <p>Every calculation is performed natively in Pandas. AI is only used for intent understanding.</p>
            </div>
            <div class="welcome-feature">
                {ICONS["sparkle"]}
                <h3>Auto-Insights</h3>
                <p>Instantly detects schemas and generates statistical summaries the moment you upload.</p>
            </div>
            <div class="welcome-feature">
                {ICONS["cpu"]}
                <h3>Explainable AI</h3>
                <p>Full transparency. View the exact execution timeline and rows scanned for every query.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True
    )

if st.session_state.uploaded_file is None:
    st.stop()

# Clear placeholder
placeholder.empty()

# --------------------------------------------------------------------------
# STATE 2: WORKSPACE MODE
# --------------------------------------------------------------------------
try:
    file_bytes = st.session_state.uploaded_file.getvalue()
    df = cached_load_dataframe(file_bytes)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    if st.button("Reset"):
        st.session_state.pop("uploaded_file")
        st.rerun()
    st.stop()

df_h = _df_hash(df)
schema = cached_schema(df_h, df)
profile = cached_profile(df_h, df)

# HEADER
st.markdown(
    f'''
    <div class="workspace-header">
        <div class="workspace-title">
            <div class="ws-icon">{ICONS["database"]}</div>
            <div>
                <div class="ws-name">{getattr(st.session_state.uploaded_file, "name", "Dataset")}</div>
                <div class="ws-meta">
                    <span>{ICONS["rows"]} {profile["rows"]:,} rows</span>
                    <span>{ICONS["cols"]} {profile["columns"]} columns</span>
                </div>
            </div>
        </div>
        <div>
            <!-- Header Actions can go here -->
        </div>
    </div>
    ''', unsafe_allow_html=True
)

# KPI BENTO
html_kpi = '<div class="kpi-grid">'
kpis = [
    ("Total Records", f"{profile['rows']:,}", "accent-blue", ICONS["rows"]),
    ("Features", f"{profile['columns']}", "accent-purple", ICONS["cols"]),
    ("Missing Data", f"{profile['missing_total']:,}", "accent-amber", ICONS["alert"]),
    ("Duplicates", f"{profile['duplicate_rows']:,}", "accent-green", ICONS["check"]),
]
for label, val, color, icon in kpis:
    html_kpi += f'''
    <div class="kpi-card {color}">
        <div class="kpi-header">
            <div class="kpi-label">{label}</div>
            <div class="kpi-icon-wrap">{icon}</div>
        </div>
        <div class="kpi-val">{val}</div>
    </div>
    '''
html_kpi += '</div>'
st.markdown(html_kpi, unsafe_allow_html=True)

# LAYOUT SPLIT
col_left, col_right = st.columns([0.65, 0.35], gap="large")

with col_right:
    # SUGGESTED
    st.markdown(f'<div class="section-head">{ICONS["sparkle"]} Suggested Questions</div>', unsafe_allow_html=True)
    suggested = []
    if schema.primary_budget_col: suggested.append(f"What is the average {schema.primary_budget_col}?")
    if schema.property_type_col: suggested.append(f"How many customers want each {schema.property_type_col}?")
    if schema.location_col: suggested.append(f"Which {schema.location_col} has the highest value?")
    if schema.status_col: suggested.append(f"Breakdown of {schema.status_col}")
    
    st.markdown('<div class="chip-container">', unsafe_allow_html=True)
    clicked_question = None
    for i, q in enumerate(suggested[:4]):
        if st.button(q, key=f"sugg_{i}"):
            clicked_question = q
    
    # CSS hack to restyle buttons rendered above
    st.markdown('''
        <script>
        const btns = window.parent.document.querySelectorAll('div[data-testid="stButton"] button');
        btns.forEach(b => { if(b.innerText.includes("What") || b.innerText.includes("How") || b.innerText.includes("Which") || b.innerText.includes("Breakdown")) b.classList.add("sugg-chip"); });
        </script>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # INSIGHTS
    st.markdown(f'<div class="section-head">{ICONS["cpu"]} AI Insights</div>', unsafe_allow_html=True)
    stat_lines = cached_rule_insights(df_h, df, schema)
    insights = stat_lines
    if gemini_helper.is_configured():
        insights = cached_gemini_insights("|".join(stat_lines), stat_lines)
    
    for line in insights:
        st.markdown(f'''
        <div class="insight-box">
            <div class="insight-top">
                <span class="insight-cat">Insight</span>
            </div>
            <div class="insight-text">{line}</div>
        </div>
        ''', unsafe_allow_html=True)

with col_left:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    
    # RENDER HISTORY
    for idx, turn in enumerate(st.session_state.chat_history):
        result = turn["result"]
        
        # User
        st.markdown(f'''
        <div class="chat-turn">
            <div class="chat-user">
                <div class="msg-bubble">{turn["question"]}</div>
                <div class="avatar avatar-u">{ICONS["user"]}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # AI
        st.markdown(f'''
            <div class="chat-ai">
                <div class="avatar avatar-ai">{ICONS["bot"]}</div>
                <div class="msg-content">
                    <div class="msg-bubble-ai">{turn["summary"]}</div>
        ''', unsafe_allow_html=True)
        
        # Timeline Explainability
        intent_method = "Gemini AI" if turn["used_gemini_intent"] else "Rule-based"
        st.markdown(f'''
                    <div class="timeline">
                        <div class="timeline-title">{ICONS["cpu"]} Execution Pipeline</div>
                        <div class="tl-step">
                            <div class="tl-line"></div>
                            <div class="tl-icon">{ICONS["bot"]}</div>
                            <div class="tl-content">
                                <div class="tl-label">Intent Classification</div>
                                <div class="tl-val">Parsed via {intent_method} → <code>{result.operation}</code></div>
                            </div>
                        </div>
                        <div class="tl-step">
                            <div class="tl-line"></div>
                            <div class="tl-icon">{ICONS["database"]}</div>
                            <div class="tl-content">
                                <div class="tl-label">Pandas Processing</div>
                                <div class="tl-val">Scanned {result.rows_scanned:,} rows · {result.execution_time_ms:.1f}ms</div>
                            </div>
                        </div>
                    </div>
        ''', unsafe_allow_html=True)
        
        # Table Preview
        if result.success and result.table_result is not None and not result.table_result.empty:
            st.dataframe(result.table_result.head(TABLE_DISPLAY_ROWS), use_container_width=True)
            csv_bytes = result.table_result.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_bytes, f"result_{idx}.csv", "text/csv", key=f"dl_{idx}")
            
        # Chart
        if turn["fig"]:
            fig = turn["fig"]
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#A1A1AA"))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")

        st.markdown('</div></div></div>', unsafe_allow_html=True)
    
    # Empty State for Chat
    if not st.session_state.chat_history:
        st.markdown('<div style="margin:auto;text-align:center;color:var(--text-3);padding:40px;">Send a message to start analyzing data.</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True) # close chat history
    
    question = st.chat_input("Ask anything about your data...")
    st.markdown('</div>', unsafe_allow_html=True) # close chat container

    # Chat execution logic
    final_question = clicked_question or question
    if final_question:
        with st.spinner("Processing query..."):
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
            
            summary = None
            used_gemini_summary = False
            if result.success and gemini_helper.is_configured():
                payload = build_result_payload(result)
                summary = cached_summary(final_question, result.operation, payload)
                used_gemini_summary = summary is not None
            if summary is None:
                summary = fallback_summary(final_question, result)
                
            fig = charts.auto_visualize(result.operation, result.table_result, schema, result.columns_used) if result.success else None
            
            st.session_state.chat_history.append({
                "question": final_question,
                "summary": summary,
                "result": result,
                "fig": fig,
                "used_gemini_intent": used_gemini_intent,
                "used_gemini_summary": used_gemini_summary,
            })
            st.rerun()

# --------------------------------------------------------------------------
# DATA PREVIEW & VISUAL PROFILE (Bottom)
# --------------------------------------------------------------------------
st.markdown('<div class="table-container">', unsafe_allow_html=True)
st.markdown(f'<div class="section-head">{ICONS["database"]} Dataset Preview</div>', unsafe_allow_html=True)
st.dataframe(df.head(DATA_PREVIEW_ROWS), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Overview charts
overview_figs = charts.profile_overview_charts(df, schema)
if overview_figs:
    st.markdown('<div style="margin-top:32px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-head">{ICONS["sparkle"]} Visual Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i, (name, fig) in enumerate(overview_figs.items()):
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#A1A1AA"))
        with (c1 if i%2==0 else c2):
            st.markdown(f'<div class="chart-box"><div class="chart-box-title">{fig.layout.title.text if fig.layout.title else name}</div>', unsafe_allow_html=True)
            fig.layout.title = None
            st.plotly_chart(fig, use_container_width=True, key=f"prof_{name}")
            st.markdown('</div>', unsafe_allow_html=True)
