"""
query_engine.py
---------------
Backward-compatible re-export shim.

The actual implementation lives in:
- operations.py  — QueryEngine class, rule-based parser, helpers
- sandbox.py     — execute_sandboxed, run_dynamic_query
"""
from operations import (  # noqa: F401
    QueryEngine,
    QueryResult,
    StepTrace,
    merge_follow_up_conditions,
    rule_based_intent,
    validate_chain_intent,
)
from sandbox import execute_sandboxed, run_dynamic_query  # noqa: F401

