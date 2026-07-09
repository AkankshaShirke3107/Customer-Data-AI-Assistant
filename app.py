"""
app.py
------
Customer Data AI Assistant
UI: Premium redesign for real estate sales professionals.
    Inspired by ChatGPT Enterprise · Notion AI · Vercel · Stripe.
Backend: 100% unchanged — query_engine, gemini_helper, charts, exports,
         conversational memory, intent classification, Pandas execution.
"""
from __future__ import annotations
import html as html_mod
import io
import logging
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import charts
import gemini_helper
from charts import CHART_CONFIG
from config import (
    CACHE_TTL_SECONDS, DATA_PREVIEW_ROWS, LOG_FORMAT, LOG_LEVEL,
    MAX_CHAT_HISTORY, TABLE_DISPLAY_ROWS, ENABLE_DYNAMIC_CODE_GEN,
)
from matching import build_value_index
from query_engine import (
    QueryEngine, merge_follow_up_conditions, rule_based_intent, validate_chain_intent, run_dynamic_query
)
from utils import detect_schema, load_dataframe, profile_dataset, rule_based_insights, validate_upload

load_dotenv()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Customer Data AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Customer Data AI Assistant — Chat with your data using natural language."},
)

# ── CSS ────────────────────────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "style.css"), "r") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


def _e(text: str) -> str:
    """HTML-escape to prevent XSS."""
    return html_mod.escape(str(text))


# ── CACHED BACKEND HELPERS (unchanged) ────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """Load an Excel or CSV file from raw bytes into a DataFrame.

    Cached by Streamlit on ``file_bytes`` content.  Delegates to
    ``utils.load_dataframe``.  Does not mutate ``st.session_state``.
    """
    return load_dataframe(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def cached_profile(_h: str, df: pd.DataFrame) -> dict:
    """Return a dataset profile dict (row/column counts, per-column stats).

    Keyed on the content hash ``_h`` so the profile is recomputed only
    when the DataFrame changes.  Does not mutate ``st.session_state``.
    """
    return profile_dataset(df)

@st.cache_data(show_spinner=False)
def cached_schema(_h: str, df: pd.DataFrame):
    """Detect and return a ``DatasetSchema`` for the given DataFrame.

    Identifies semantic column roles (budget, location, status, etc.)
    via keyword matching.  Keyed on content hash ``_h``.
    Does not mutate ``st.session_state``.
    """
    return detect_schema(df)

@st.cache_data(show_spinner=False)
def cached_value_index(_h: str, df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    """Build the fuzzy-matching value index for categorical columns.

    Returns ``{col: {normalized_value: canonical_value}}`` used by the
    query engine's condition-application layer.  Keyed on content hash.
    Does not mutate ``st.session_state``.
    """
    return build_value_index(df, categorical_cols)

@st.cache_data(show_spinner=False)
def cached_rule_insights(_h: str, df: pd.DataFrame, _schema) -> list[str]:
    """Generate rule-based statistical insight strings for the dataset.

    Used as input to the AI Insights panel and as a Gemini fallback.
    Keyed on content hash.  Does not mutate ``st.session_state``.
    """
    return rule_based_insights(df, _schema)

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_gemini_insights(key: str, lines: list[str]) -> list[str]:
    """Enhance rule-based insight lines via Gemini LLM rephrasing.

    Falls back gracefully if Gemini is unavailable (caller uses raw lines).
    Cached with a TTL of ``CACHE_TTL_SECONDS``.
    Does not mutate ``st.session_state``.
    """
    return gemini_helper.generate_insights(lines)

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_intent(q: str, cols: list[str], schema_dict: dict) -> dict | None:
    """Classify a user question into a structured intent via Gemini.

    Returns a ``RootIntentModel`` (or dict) on success, ``None`` on any
    exception (caller falls back to the rule-based parser).
    Cached with a TTL of ``CACHE_TTL_SECONDS``.
    Does not mutate ``st.session_state``.
    """
    try:
        return gemini_helper.understand_intent(q, cols, schema_dict)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_summary(q: str, op: str, payload: dict) -> str | None:
    """Ask Gemini to phrase a computed result as a natural-language summary.

    Returns ``None`` on any exception (caller uses ``fallback_summary``).
    Cached with a TTL of ``CACHE_TTL_SECONDS``.
    Does not mutate ``st.session_state``.
    """
    try:
        return gemini_helper.summarize_result(q, op, payload)
    except Exception:
        return None


def _df_hash(df: pd.DataFrame) -> str:
    """Compute a lightweight content hash for Streamlit cache keying.

    Combines row count, column count, column names, dtypes, and a
    3-row JSON sample.  Intentionally cheap — avoids hashing the full
    DataFrame.  Does not mutate ``st.session_state``.
    """
    sample = df.head(3).to_json()
    return f"{len(df)}_{df.shape[1]}_{hash(tuple(df.columns))}_{hash(tuple(df.dtypes))}_{hash(sample)}"


def build_result_payload(result) -> dict:
    """Convert a ``QueryResult`` into a JSON-safe dict for Gemini summarization.

    Includes the operation name, scalar result, and an optional 15-row
    table preview with row count.  Does not mutate ``st.session_state``.
    """
    payload = {"operation": result.operation, "scalar_result": result.scalar_result}
    if result.table_result is not None and not result.table_result.empty:
        payload["table_preview"] = result.table_result.head(15).to_dict(orient="records")
        payload["row_count"] = len(result.table_result)
    return payload


def fallback_summary(question: str, result) -> str:
    """Generate a template-based summary when Gemini is unavailable.

    Handles three cases: failed queries (error message), scalar results
    (formatted value), and table results (row count).  Used as the final
    fallback after ``cached_summary`` returns ``None``.
    Does not mutate ``st.session_state``.
    """
    if not result.success:
        return f"I wasn't able to process that request. {result.error}"
    if result.scalar_result is not None and (
        result.table_result is None
        or result.operation in {"count", "sum", "average", "median", "min", "max", "distinct_count", "missing"}
    ):
        val = result.scalar_result
        if isinstance(val, float):
            val = round(val, 2)
        return f"The answer is **{val}**."
    if result.table_result is not None:
        if result.operation == "groupby":
            return f"Here is the breakdown by category ({len(result.table_result)} groups found)."
        return f"I found **{len(result.table_result)} customers** matching your criteria."
    return "Done! Here are the results based on your dataset."


# ── SESSION STATE ──────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# ── SIDEBAR (settings only) ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if gemini_helper.is_configured():
        st.success("✓ Gemini AI Connected")
    else:
        st.warning("Gemini API key missing")
        _key = st.text_input("API Key", type="password")
        if _key:
            os.environ["GEMINI_API_KEY"] = _key
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════
_placeholder = st.empty()

with _placeholder.container():
    st.markdown(
        '''
        <div class="welcome-hero">
            <div class="welcome-pill">✦ AI-Powered Customer Intelligence</div>
            <h1 class="welcome-title">Chat with your customer<br>data in plain English.</h1>
            <p class="welcome-subtitle">
                Upload your sales data and get instant answers — no formulas, no filters,
                no spreadsheet skills required.
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        uploaded = st.file_uploader(
            "Drop your dataset",
            type=["xlsx", "xls", "csv"],
            label_visibility="collapsed",
            help="Excel (.xlsx, .xls) or CSV — up to 50 MB",
        )
        if uploaded:
            try:
                validate_upload(uploaded)
            except ValueError as ve:
                st.error(str(ve))
                st.stop()
            st.session_state.uploaded_file = uploaded
            st.rerun()

        if st.button("Try with Sample Dataset →", use_container_width=True):
            _sample = os.path.join(os.path.dirname(__file__), "data", "sample_leads.xlsx")
            with open(_sample, "rb") as _sf:
                st.session_state.uploaded_file = io.BytesIO(_sf.read())
                st.session_state.uploaded_file.name = "sample_leads.xlsx"
            st.rerun()

    st.markdown(
        '''
        <div class="welcome-features">
            <div class="welcome-feature">
                <div class="wf-icon">💬</div>
                <h3>Ask in Plain English</h3>
                <p>Just type your question. No formulas, no filters, no training needed.</p>
            </div>
            <div class="welcome-feature">
                <div class="wf-icon">📊</div>
                <h3>Instant Visual Insights</h3>
                <p>Charts and summaries generated automatically from your data.</p>
            </div>
            <div class="welcome-feature">
                <div class="wf-icon">✓</div>
                <h3>Always Accurate</h3>
                <p>Every answer is computed directly from your uploaded file — never guessed.</p>
            </div>
        </div>
        <div style="text-align:center;margin-top:20px;color:var(--text-4);font-size:12px;">
            Supports Excel (.xlsx, .xls) and CSV — up to 50 MB
        </div>
        ''',
        unsafe_allow_html=True,
    )

if st.session_state.uploaded_file is None:
    st.stop()

_placeholder.empty()


# ══════════════════════════════════════════════════════════════════════════
# WORKSPACE — load data & compute metadata
# ══════════════════════════════════════════════════════════════════════════
try:
    _file_bytes = st.session_state.uploaded_file.getvalue()
    df = cached_load_dataframe(_file_bytes)
except Exception as _e_load:
    st.error(f"Couldn't load the file. Please try a different dataset. ({_e_load})")
    if st.button("↩ Start Over"):
        st.session_state.pop("uploaded_file", None)
        st.rerun()
    st.stop()

df_h   = _df_hash(df)
schema = cached_schema(df_h, df)
profile = cached_profile(df_h, df)

_fname = _e(getattr(st.session_state.uploaded_file, "name", "Dataset"))

# Compute average budget (business-friendly)
_avg_fmt        = ""
_avg_budget_raw = None
_avg_stat_html  = ""
if schema.primary_budget_col and schema.primary_budget_col in df.columns:
    try:
        _avg_budget_raw = df[schema.primary_budget_col].mean()
        if _avg_budget_raw > 100000:
            _avg_fmt = f"₹{_avg_budget_raw/100000:.1f}L"
        else:
            _avg_fmt = f"{_avg_budget_raw:,.0f}"
        _avg_stat_html = f'''
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <div class="hero-stat-value">{_avg_fmt}</div>
            <div class="hero-stat-label">Avg {_e(schema.primary_budget_col)}</div>
        </div>'''
    except Exception:
        pass


# ── HERO CARD ─────────────────────────────────────────────────────────────
# NOTE: _avg_stat_html is injected as trusted HTML (values are escaped via _e()).
# The entire hero-card block is one self-contained st.markdown call with
# unsafe_allow_html=True, so no orphaned tags can leak.
st.markdown(
    f'''
    <div class="hero-card">
        <div class="hero-top">
            <div class="hero-file">
                <div class="hero-file-icon">🗄️</div>
                <div>
                    <div class="hero-filename">📁 {_fname}</div>
                    <div class="hero-loaded">✓ Dataset Successfully Loaded</div>
                </div>
            </div>
            <div class="hero-status">
                <div class="hero-status-dot"></div>
                Ready for Analysis
            </div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value">{profile["rows"]:,}</div>
                <div class="hero-stat-label">Customers</div>
            </div>
            <div class="hero-divider"></div>
            <div class="hero-stat">
                <div class="hero-stat-value">{profile["columns"]}</div>
                <div class="hero-stat-label">Columns</div>
            </div>
            {_avg_stat_html}
        </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# Change dataset — plain text label (no SVG to avoid raw-text rendering)
if st.button("↩ Change Dataset", key="change_ds"):
    st.session_state.pop("uploaded_file", None)
    st.session_state.chat_history = []
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# BUILD SUGGESTED QUESTIONS (dynamic, based on schema)
# ══════════════════════════════════════════════════════════════════════════
_suggested: list[str] = []
_bc = schema.primary_budget_col
_lc = schema.location_col
_pc = schema.property_type_col
_sc = schema.status_col

if _bc:
    _suggested.append(f"What is the average {_bc}?")
    _suggested.append(f"Show customers with {_bc} above {_avg_fmt if _avg_budget_raw else 'highest value'}")
if _pc:
    _suggested.append(f"How many customers prefer each {_pc}?")
if _lc:
    _suggested.append(f"Which {_lc} has the most customers?")
if _sc:
    _suggested.append(f"Show customers by {_sc} status")
if schema.date_cols:
    _suggested.append("Show customers added this month")
_suggested.append("Who are the high-intent customers?")
_suggested.append("Give me a summary of this dataset")

# Deduplicate, cap at 6
_seen: set = set()
_unique_sugg: list[str] = []
for _q in _suggested:
    if _q not in _seen:
        _seen.add(_q)
        _unique_sugg.append(_q)
_suggested = _unique_sugg[:6]


# ══════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════
col_main, col_side = st.columns([0.65, 0.35], gap="large")


# ──────────────────────────────────────────────────────────────────────────
# RIGHT SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with col_side:

    # ── Suggested Questions ────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">✦ Suggested Questions</div>',
        unsafe_allow_html=True,
    )

    _clicked_q = None
    for _i, _q in enumerate(_suggested):
        if st.button(_q, key=f"sq_{_i}", use_container_width=True):
            _clicked_q = _q

    # ── Column Explorer (collapsed) ────────────────────────────────────
    st.markdown(
        '<div class="section-title" style="margin-top:16px">⊞ Column Explorer</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View columns & data types", expanded=False):
        _col_role = {
            schema.name_col:           "👤 Customer Name",
            schema.primary_budget_col: "💰 Budget",
            schema.location_col:       "📍 Location",
            schema.property_type_col:  "🏠 Property Type",
            schema.status_col:         "🔖 Status",
            schema.contact_col:        "📞 Contact",
        }
        for _dc in schema.date_cols:
            _col_role[_dc] = "📅 Date"
        for _ic_col in schema.id_cols:
            _col_role[_ic_col] = "🔑 ID"

        _col_rows = []
        for _cp in profile["column_profiles"]:
            _col_rows.append({
                "Column": _cp.name,
                "Type":   _col_role.get(_cp.name, "📊 Data"),
                "Sample": ", ".join(str(_v) for _v in _cp.sample_values[:2]),
            })
        st.dataframe(pd.DataFrame(_col_rows), use_container_width=True, hide_index=True)

    # ── AI Insights — single card ──────────────────────────────────────
    st.markdown(
        '<div class="section-title" style="margin-top:16px">✧ AI Insights</div>',
        unsafe_allow_html=True,
    )

    _stat_lines = cached_rule_insights(df_h, df, schema)
    _insights   = _stat_lines
    if gemini_helper.is_configured():
        _insights = cached_gemini_insights("|".join(_stat_lines), _stat_lines)

    _rows_html = ""
    for _line in _insights:
        _rows_html += f'''
        <div class="insight-row">
            <div class="insight-dot"></div>
            <div class="insight-txt">{_e(_line)}</div>
        </div>'''

    st.markdown(
        f'''
        <div class="insights-card">
            <div class="insights-head">
                <span class="insights-head-icon">✦</span>
                <span class="insights-head-title">Dataset Insights</span>
            </div>
            <div class="insights-body">{_rows_html}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────────────────────────────────
with col_main:

    # ── EMPTY STATE ────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            '''
            <div class="chat-empty">
                <div class="chat-empty-icon">✦</div>
                <div class="chat-empty-title">👋 Ask anything about your customers</div>
                <div class="chat-empty-sub">
                    Type a question below or tap a suggestion to get started instantly.
                </div>
                <div class="chat-empty-label">Try asking</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # Clickable example chips — rendered as Streamlit buttons with CSS targeting
        _ex_questions = [
            "Customers with budget above ₹90L",
            "Show all 3BHK buyers",
            "Average budget",
            "High-intent customers",
            "Which location has most leads?",
        ]
        # Use columns layout to create a centered chip grid
        _chip_col1, _chip_col2 = st.columns(2)
        for _ei, _eq in enumerate(_ex_questions):
            _col = _chip_col1 if _ei % 2 == 0 else _chip_col2
            with _col:
                if st.button(_eq, key=f"ex_{_ei}", use_container_width=True):
                    _clicked_q = _eq

    # ── CHAT HISTORY ───────────────────────────────────────────────────
    for _idx, _turn in enumerate(st.session_state.chat_history):
        _result = _turn["result"]

        # User message
        st.markdown(
            f'''
            <div class="user-row">
                <div class="user-bubble">{_e(_turn["question"])}</div>
                <div class="user-avatar">👤</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # AI response
        st.markdown(
            f'''
            <div class="ai-row">
                <div class="ai-avatar">✦</div>
                <div class="ai-bubble-wrap">
                    <div class="response-answer">{_e(_turn["summary"])}</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # Step trace (multi-step chains only)
        if _result.step_trace:
            with st.expander(f"🔗 Execution Pipeline — {len(_result.step_trace)} steps", expanded=False):
                for _st in _result.step_trace:
                    st.markdown(
                        f"**Step {_st.step_number}** · `{_e(_st.operation)}` · "
                        f"{_st.rows_before} → {_st.rows_after} rows · "
                        f"{_st.execution_time_ms}ms",
                    )
                    st.caption(_e(_st.explanation))
                    
        # Fuzzy match corrections
        if hasattr(_result, 'fuzzy_matches') and _result.fuzzy_matches:
            # We only surface below-perfect confidence matches to the user
            _corrections = [fm for fm in _result.fuzzy_matches if fm["method"] != "exact" and fm["method"] != "normalized"]
            if _corrections:
                for _corr in _corrections:
                    st.caption(f"💡 Interpreted '{_e(_corr['original'])}' as '{_e(_corr['matched'])}'")

        # Dynamic Code execution display
        if _result.execution_path == "dynamic" and _result.dynamic_code:
            with st.expander("⚡ Dynamic Code Execution", expanded=False):
                st.code(_result.dynamic_code, language="python")
                if _result.dynamic_retry_count > 0:
                    st.caption(f"Self-corrected after {_result.dynamic_retry_count} retry(s)")

        # Table result
        if _result.success and _result.table_result is not None and not _result.table_result.empty:
            st.dataframe(_result.table_result.head(TABLE_DISPLAY_ROWS), use_container_width=True)
            _csv = _result.table_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download CSV",
                _csv,
                f"results_{_idx + 1}.csv",
                "text/csv",
                key=f"dl_{_idx}",
            )

        # Empty result
        elif _result.success and (
            _result.scalar_result == 0
            or (_result.table_result is not None and _result.table_result.empty)
        ):
            st.markdown(
                '''
                <div class="no-results">
                    <div class="no-results-emoji">🔍</div>
                    <div class="no-results-title">No results found</div>
                    <div class="no-results-hint">
                        Try rephrasing — for example, check spelling of city names or property types.
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

        # Chart
        if _turn["fig"]:
            st.plotly_chart(_turn["fig"], use_container_width=True, key=f"chart_{_idx}")

        # Turn separator
        st.markdown('<div class="chat-turn-divider"></div>', unsafe_allow_html=True)

    # ── Export Chat ────────────────────────────────────────────────────
    if st.session_state.chat_history:
        def _build_export() -> str:
            """Compile the full chat history into a Markdown report.

            Reads ``st.session_state.chat_history`` (list of turn dicts)
            and ``st.session_state.uploaded_file.name``.
            Does not mutate ``st.session_state``.
            """
            _lines = [f"# Chat Export — {getattr(st.session_state.uploaded_file, 'name', 'Dataset')}\n"]
            for _i, _t in enumerate(st.session_state.chat_history, 1):
                _r = _t["result"]
                _lines.append(f"## Q{_i}: {_t['question']}")
                _lines.append(f"**Answer:** {_t['summary']}")
                if _r.table_result is not None and not _r.table_result.empty:
                    _lines.append(_r.table_result.head(20).to_markdown(index=False))
                _lines.append("")
            return "\n".join(_lines)

        try:
            _md = _build_export().encode("utf-8")
            st.download_button(
                "⬇ Export Full Chat",
                _md,
                "chat_export.md",
                "text/markdown",
                key="export_chat",
                use_container_width=True,
            )
        except Exception:
            pass

    # ── Chat Input ─────────────────────────────────────────────────────
    _question = st.chat_input("Ask anything about your customer data…")

    # ── QUERY EXECUTION (backend 100% unchanged) ───────────────────────
    _final_q = _clicked_q or _question
    if _final_q:
        with st.spinner("Analysing your data…"):
            _cols       = list(df.columns)
            _schema_d   = schema.to_dict()
            _intent     = None
            _used_gem   = False

            if gemini_helper.is_configured():
                _intent   = cached_intent(_final_q, _cols, _schema_d)
                _used_gem = _intent is not None
            if _intent is None:
                _intent = rule_based_intent(_final_q, schema, df)

            # Conversational memory: merge previous filters
            if st.session_state.chat_history:
                _prev = st.session_state.chat_history[-1]
                _prev_intent = _prev.get("intent")
                _prev_cond = getattr(_prev_intent, "conditions", []) if _prev_intent else []
                _intent = merge_follow_up_conditions(_intent, _prev_cond)
            _value_index = cached_value_index(_df_hash(df), df, schema.categorical_cols)

            _engine  = QueryEngine(df, schema, value_index=_value_index)
            if _intent.steps:
                try:
                    _validated_steps = validate_chain_intent(_intent)
                except ValueError as _ve:
                    logger.warning("Chain validation failed: %s", _ve)
                    _intent = rule_based_intent(_final_q, schema, df)
                if _intent.steps:
                    _result = _engine.execute_chain(_intent.steps)
                else:
                    _result = _engine.execute(_intent)
            else:
                _result  = _engine.execute(_intent)

            # Dynamic code generation fallback
            if (not _result.success
                and ENABLE_DYNAMIC_CODE_GEN
                and gemini_helper.is_configured()
                and "Unrecognized operation" in (_result.error or "")):
                _result = run_dynamic_query(_final_q, df, schema, value_index=_value_index)

            _summary      = None
            _used_gem_sum = False
            if _result.success and gemini_helper.is_configured():
                _payload      = build_result_payload(_result)
                _summary      = cached_summary(_final_q, _result.operation, _payload)
                _used_gem_sum = _summary is not None
            if _summary is None:
                _summary = fallback_summary(_final_q, _result)

            _fig = (
                charts.auto_visualize(
                    _result.operation, _result.table_result, schema, _result.columns_used
                )
                if _result.success
                else None
            )

            # Cap the dataframe memory footprint for historical turns
            if _result.success and _result.table_result is not None:
                _result.table_result = _result.table_result.head(TABLE_DISPLAY_ROWS).copy()

            st.session_state.chat_history.append({
                "question":           _final_q,
                "summary":            _summary,
                "result":             _result,
                "intent":             _intent,
                "fig":                _fig,
                "used_gemini_intent": _used_gem,
                "used_gemini_summary":_used_gem_sum,
            })

            if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# DATASET PREVIEW  (collapsed by default)
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
with st.expander(f"Dataset Preview — first {DATA_PREVIEW_ROWS} rows", expanded=False):
    st.dataframe(df.head(DATA_PREVIEW_ROWS), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# INTERACTIVE ANALYTICS  (all charts preserved)
# ══════════════════════════════════════════════════════════════════════════
_overview_figs = charts.profile_overview_charts(df, schema)
if _overview_figs:
    st.markdown(
        '<div class="analytics-header">📊 Interactive Analytics</div>',
        unsafe_allow_html=True,
    )
    _c1, _c2 = st.columns(2)
    for _ai, (_nm, _fig) in enumerate(_overview_figs.items()):
        _fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9494B8"),
        )
        _title_txt = (
            _fig.layout.title.text
            if _fig.layout.title and _fig.layout.title.text
            else _nm
        )
        with (_c1 if _ai % 2 == 0 else _c2):
            st.markdown(
                f'<div class="chart-card"><div class="chart-card-title">{_e(_title_txt)}</div>',
                unsafe_allow_html=True,
            )
            _fig.layout.title = None
            st.plotly_chart(_fig, use_container_width=True, key=f"ov_{_nm}")
            st.markdown('</div>', unsafe_allow_html=True)
