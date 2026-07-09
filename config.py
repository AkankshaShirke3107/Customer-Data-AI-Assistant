"""
config.py
---------
Central configuration for the Customer Data AI Assistant.

All magic numbers, limits, and tunables live here so they can be
changed in one place without hunting through application code.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# File upload limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: set[str] = {".xlsx", ".xls", ".csv"}

# ---------------------------------------------------------------------------
# Chat / session limits
# ---------------------------------------------------------------------------
MAX_CHAT_HISTORY: int = 50  # oldest turns are evicted when this is exceeded

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SECONDS: int = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30"))
GEMINI_MAX_RETRIES: int = 2
CACHE_TTL_SECONDS: int = 3600  # 1 hour for Gemini response caches

# ---------------------------------------------------------------------------
# UI / display
# ---------------------------------------------------------------------------
DATA_PREVIEW_ROWS: int = 20
TABLE_DISPLAY_ROWS: int = 200
DEFAULT_CHART_MARGIN: dict = dict(l=10, r=10, t=50, b=10)

# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------
OPERATIONS: set[str] = {
    "count", "sum", "average", "median", "min", "max", "filter", "sort",
    "groupby", "topn", "bottomn", "between", "greater_than", "less_than",
    "unique", "distinct_count", "describe", "list", "date_filter",
    "missing",
}
MAX_QUERY_STEPS: int = 5  # upper bound on chained query steps

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s"
