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
spreadsheet into a single JSON object describing which pandas operation to
run.

The dataframe has these columns: {columns}
Semantic role hints (best guess, may be incomplete): {schema}

Return ONLY a single valid JSON object (no markdown fences, no prose) with
this shape:
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

Rules:
- "column" must be copied as close as possible from the real column list above; if unsure, guess the closest match.
- If the question implies multiple filters (e.g. "2BHK in Pune"), put both as separate entries in "conditions".
- If the question asks for a count, use "count". If it asks to "list" or "show me" records, use "list".
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


def _validate_intent(intent: dict) -> dict:
    """Sanitize and validate a parsed intent dict before execution.

    - Ensures `operation` is in the allowed set.
    - Strips any unexpected top-level keys.
    """
    op = (intent.get("operation") or "").lower().strip()
    if op not in OPERATIONS:
        raise ValueError(
            f"Gemini returned invalid operation '{op}'. "
            f"Allowed: {sorted(OPERATIONS)}"
        )
    intent["operation"] = op
    # Allow only known keys through
    allowed_keys = {
        "operation", "column", "group_by", "agg_column", "agg_func",
        "value", "value2", "n", "ascending", "conditions",
    }
    return {k: v for k, v in intent.items() if k in allowed_keys}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def understand_intent(question: str, columns: list[str], schema_dict: dict) -> dict:
    """Ask Gemini to classify the question into a structured intent dict.

    Raises GeminiUnavailableError if the API/key isn't configured, and
    ValueError if Gemini's response can't be parsed as valid JSON – in
    both cases the caller should fall back to the rule-based parser in
    query_engine.py.
    """
    client = _get_client()
    prompt = _INTENT_SYSTEM_PROMPT.format(
        columns=json.dumps(columns),
        schema=json.dumps(schema_dict),
        ops=json.dumps(_ALLOWED_OPS),
    )
    full_prompt = f"{prompt}\n\nUser question: {question}\nJSON:"
    raw_text = _call_with_retry(client, full_prompt)
    intent = _extract_json(raw_text)
    intent = _validate_intent(intent)
    logger.info("Parsed intent: operation=%s", intent.get("operation"))
    return intent


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


def is_configured() -> bool:
    """Check if the Gemini SDK and API key are available."""
    return _GENAI_AVAILABLE and bool(os.environ.get("GEMINI_API_KEY"))
