"""
gemini_helper.py
----------------
ALL calls to Google Gemini live here, and ONLY here. Gemini is used for
exactly three things in this application:

1. `understand_intent`   – turn a natural-language question into a
                            structured JSON "intent" dict (operation,
                            column, filters, ...). Gemini never computes
                            the answer itself.
2. `summarize_result`     – turn an already-computed pandas result into
                            a friendly natural-language sentence.
3. `generate_insights`    – turn already-computed pandas statistics into
                            friendly bullet-point insights.

Gemini is never shown the full raw dataset and is never asked to
"answer" a question directly – it only ever sees already-computed
numbers/tables and rephrases them, or it maps a question to an
operation name from a fixed vocabulary. This is what prevents
hallucination.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from config import (
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
    MAX_QUERY_STEPS,
    OPERATIONS,
)

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

_ALLOWED_OPS = sorted(OPERATIONS)

# ---------------------------------------------------------------------------
# SDK configuration guard – configure only once per process unless key changes
# ---------------------------------------------------------------------------
_SDK_CONFIGURED: bool = False
_LAST_API_KEY: str = ""


def _ensure_sdk_configured(api_key: str) -> None:
    """Configure the Gemini SDK when the API key changes or is set for the first time."""
    global _SDK_CONFIGURED, _LAST_API_KEY
    if not _SDK_CONFIGURED or api_key != _LAST_API_KEY:
        genai.configure(api_key=api_key)
        _SDK_CONFIGURED = True
        _LAST_API_KEY = api_key
        logger.info("Gemini SDK configured (model=%s)", GEMINI_MODEL)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_INTENT_SYSTEM_PROMPT = """You are a strict intent-classification engine for a
data analysis tool. You NEVER answer questions yourself and you NEVER invent
numbers. Your only job is to map a user's natural language question about a
spreadsheet into a JSON object describing which pandas operation(s) to run.

The dataframe has these columns: {columns}
Semantic role hints (best guess, may be incomplete): {schema}

--- Single-operation format (use when the question needs one operation) ---

Return a single JSON object with this shape:
{{
  "operation": one of {ops},
  "column": "<column name or null>",
  "group_by": "<column name or null>",
  "agg_column": "<column name or null>",
  "agg_func": "mean|sum|count|min|max|median or null",
  "value": <number or string or null>,
  "value2": <number or string or null>,
  "n": <integer or null>,
  "ascending": <true|false>,
  "conditions": [
    {{"column": "<col>", "op": "eq|neq|gt|gte|lt|lte|between|contains", "value": <val>, "value2": <val or null>}}
  ]
}}

--- Multi-step format (use ONLY when the question chains multiple sequential
    operations, e.g. "filter X then sort by Y then top 5") ---

Return:
{{
  "steps": [
    {{<single-operation object as above>}},
    {{<single-operation object as above>}},
    ...
  ]
}}

Each element of "steps" has the exact same shape as the single-operation
format.  The result of step i is fed as the working dataset for step i+1.
Use at most {max_steps} steps.

Rules:
- Use the multi-step format ONLY when the question genuinely requires
  chaining.  For simple questions (e.g. "how many customers?"), always
  return the single-operation format.
- "column" must be copied as close as possible from the real column list above; if unsure, guess the closest match.
- If the question implies multiple filters (e.g. "2BHK in Pune"), put both as separate entries in "conditions".
- If the question asks for a count, use "count". If it asks to "list" or "show me" records, use "list".
- If the question asks to count occurrences PER category or asks for the most/least common category (e.g., "how many customers per city", "most popular location", "how many prefer each type"), use "groupby" with agg_func="count".
- If the question compares a group (e.g. "average budget by city"), use "groupby".
- Never fabricate a numeric answer yourself; only describe the operation.
- Respond with JSON only.
"""

_SUMMARY_SYSTEM_PROMPT = """You are a data assistant. You will be given:
- the user's original question
- the exact operation that was executed
- the exact numeric/tabular result computed by pandas (already correct - do not change it)

Write a short (1-4 sentence), friendly, natural-language answer to the
user's question using ONLY the numbers/values given to you. Do not invent
any numbers that are not present in the provided result. If the result is
empty, say so clearly and suggest the user rephrase.
"""

_INSIGHTS_SYSTEM_PROMPT = """You are a data assistant. You will be given a list
of already-computed factual statistics about a dataset. Rewrite them as 3-6
punchy, business-friendly bullet insights. Do NOT add any statistic, number,
or claim that is not already present in the input. Return plain text, one
insight per line, no markdown bullets needed (the UI adds its own bullets).
"""


class GeminiUnavailableError(Exception):
    """Raised when the Gemini SDK or API key is not available."""
    pass


def _get_client():
    """Return a configured GenerativeModel, configuring the SDK once."""
    if not _GENAI_AVAILABLE:
        raise GeminiUnavailableError("google-generativeai package is not installed.")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiUnavailableError("GEMINI_API_KEY is not set in the environment.")
    _ensure_sdk_configured(api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def _call_with_retry(client, prompt: str) -> str:
    """Call Gemini with timeout awareness and retry logic."""
    last_exc: Exception | None = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            t0 = time.time()
            response = client.generate_content(
                prompt,
                request_options={"timeout": GEMINI_TIMEOUT_SECONDS},
            )
            latency = time.time() - t0
            text = (response.text or "").strip()
            logger.info(
                "Gemini call OK (attempt=%d, latency=%.2fs, chars=%d)",
                attempt, latency, len(text),
            )
            return text
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Gemini call failed (attempt=%d/%d): %s",
                attempt, GEMINI_MAX_RETRIES, exc,
            )
    raise last_exc  # type: ignore[misc]


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a possibly-fenced / chatty LLM response."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Grab the first {...} block if there's still stray text around it
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)




# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
from models import RootIntentModel
from pydantic import ValidationError

def understand_intent(question: str, columns: list[str], schema_dict: dict) -> RootIntentModel:
    """Ask Gemini to classify the question into a structured intent model.

    Raises GeminiUnavailableError if the API/key isn't configured, and
    ValueError if Gemini's response can't be parsed as valid JSON – in
    both cases the caller should fall back to the rule-based parser in
    query_engine.py.
    """
    client = _get_client()
    base_prompt = _INTENT_SYSTEM_PROMPT.format(
        columns=json.dumps(columns),
        schema=json.dumps(schema_dict),
        ops=json.dumps(_ALLOWED_OPS),
        max_steps=MAX_QUERY_STEPS,
    )
    
    current_prompt = f"{base_prompt}\n\nUser question: {question}\nJSON:"
    last_exc: Exception | None = None

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            t0 = time.time()
            response = client.generate_content(
                current_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RootIntentModel
                ),
                request_options={"timeout": GEMINI_TIMEOUT_SECONDS},
            )
            latency = time.time() - t0
            text = (response.text or "").strip()
            logger.info("Gemini call OK (attempt=%d, latency=%.2fs, chars=%d)", attempt, latency, len(text))
            
            parsed_json = _extract_json(text)
            intent = RootIntentModel.model_validate(parsed_json)
            logger.info("Parsed intent: operation=%s, steps=%d", intent.operation, len(intent.steps) if intent.steps else 0)
            return intent
            
        except (json.JSONDecodeError, ValidationError) as exc:
            last_exc = exc
            text = text if 'text' in locals() else ""
            logger.warning("Gemini parsing/validation failed (attempt=%d/%d): %s", attempt, GEMINI_MAX_RETRIES, exc)
            current_prompt = f"{base_prompt}\n\nUser question: {question}\nFailed JSON Output:\n{text}\nValidation Error: {str(exc)}\nPlease fix the JSON to match the schema."
        except Exception as exc:
            last_exc = exc
            logger.warning("Gemini call failed (attempt=%d/%d): %s", attempt, GEMINI_MAX_RETRIES, exc)
            
    raise ValueError(f"Failed to generate valid intent after {GEMINI_MAX_RETRIES} attempts: {last_exc}") from last_exc


def summarize_result(question: str, operation: str, result_payload: dict) -> str:
    """Ask Gemini to phrase the already-computed result as a sentence."""
    client = _get_client()
    prompt = (
        f"{_SUMMARY_SYSTEM_PROMPT}\n\n"
        f"User question: {question}\n"
        f"Operation executed: {operation}\n"
        f"Computed result (ground truth, do not alter numbers): "
        f"{json.dumps(result_payload, default=str)}\n\nAnswer:"
    )
    return _call_with_retry(client, prompt)


def generate_insights(stats_lines: list[str]) -> list[str]:
    """Ask Gemini to rewrite pre-computed stat strings as polished insights."""
    client = _get_client()
    prompt = (
        f"{_INSIGHTS_SYSTEM_PROMPT}\n\nFacts:\n"
        + "\n".join(f"- {s}" for s in stats_lines)
    )
    text = _call_with_retry(client, prompt)
    lines = [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]
    return lines or stats_lines


# ---------------------------------------------------------------------------
# Dynamic Pandas Code Generation
# ---------------------------------------------------------------------------
_CODE_GEN_SYSTEM_PROMPT = """You are a pandas code generator. Given a user question
and a DataFrame schema, write ONLY executable Python code that operates on a
DataFrame named `df`.

Rules:
- Use ONLY `df`, `pd` (pandas), `np` (numpy), and safe builtins (len, sum, min,
  max, round, abs, sorted, list, dict, str, int, float, bool, isinstance, range,
  enumerate, zip, map, filter).
- Do NOT use import statements, file I/O, network calls, eval, exec, __import__,
  open, compile, globals, locals, getattr, setattr, or delattr.
- Do NOT use os, sys, subprocess, signal, or any system access.
- For string comparisons, you may compare against `df['<col>'].astype(str).str.strip().str.lower()` for case/whitespace-insensitive matching.
- Return ONLY the code. No prose, no markdown fences, no explanation.
- The code should be a single expression OR a short block that assigns the final
  result to a variable named `result`.
- For table results, produce a DataFrame. For scalar results, produce the value.
- Column names (use exactly as listed): {columns}
- Column data types: {dtypes}
- Categorical canonical values (if available, use these exact strings): {categories}
- Sample data (first 3 rows): {sample}
"""

_CODE_CORRECTION_PROMPT = """The following pandas code failed with an error.
Write corrected code that fixes this error. Follow the same rules as before:
use only `df`, `pd`, `np`, and safe builtins. Return ONLY the corrected code,
no explanation, no markdown fences.

Original question: {question}
DataFrame columns: {columns}
Failed code:
{code}
Error: {error}
"""


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences that Gemini sometimes adds despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def generate_pandas_code(question: str, df_schema: dict[str, Any], value_index: dict | None = None) -> str:
    """Ask Gemini to generate executable pandas code for a user question.

    The prompt includes column names, dtypes, and 2-3 sample rows (never full
    PII beyond what's already in-memory). Raw row-level data is NOT sent.

    Returns the generated code string ready for sandboxed execution.

    Raises GeminiUnavailableError if the API is not configured.
    """
    client = _get_client()
    
    # Format canonical values for the prompt
    canonical_vals = {}
    if value_index:
        for col, col_idx in value_index.items():
            canonical_vals[col] = list(col_idx.values())
            
    prompt = _CODE_GEN_SYSTEM_PROMPT.format(
        columns=json.dumps(df_schema.get("columns", [])),
        dtypes=json.dumps(df_schema.get("dtypes", {})),
        categories=json.dumps(canonical_vals),
        sample=json.dumps(df_schema.get("sample_rows", []), default=str),
    )
    full_prompt = f"{prompt}\n\nUser question: {question}\nCode:"
    raw_text = _call_with_retry(client, full_prompt)
    return _strip_code_fences(raw_text)


def generate_corrected_code(
    question: str,
    failed_code: str,
    error_msg: str,
    df_schema: dict[str, Any],
) -> str:
    """Ask Gemini to correct previously-failed pandas code.

    Sends the original question, the failed code, and a sanitized error
    message. No raw dataset rows are included.

    Returns the corrected code string.

    Raises GeminiUnavailableError if the API is not configured.
    """
    client = _get_client()
    prompt = _CODE_CORRECTION_PROMPT.format(
        question=question,
        columns=json.dumps(df_schema.get("columns", [])),
        code=failed_code,
        error=error_msg,
    )
    raw_text = _call_with_retry(client, prompt)
    return _strip_code_fences(raw_text)


def is_configured() -> bool:
    """Check if the Gemini SDK and API key are available."""
    return _GENAI_AVAILABLE and bool(os.environ.get("GEMINI_API_KEY"))

