"""
sandbox.py
----------
Sandboxed dynamic code execution for LLM-generated pandas code.

Provides `execute_sandboxed` for running generated code in a restricted
environment with timeout control, and `run_dynamic_query` which orchestrates
the generate-execute-retry loop using Gemini for code generation and
self-correction.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pandas as pd

from config import (
    DYNAMIC_CODE_MAX_RETRIES,
    DYNAMIC_RESULT_MAX_ROWS,
    SANDBOX_TIMEOUT_SECONDS,
)
from operations import QueryResult
from utils import DatasetSchema

logger = logging.getLogger(__name__)


def execute_sandboxed(code: str, df: pd.DataFrame, timeout: int = 5) -> dict:
    """Execute generated pandas code in a restricted sandbox.
    
    Args:
        code: The python code to execute.
        df: The dataframe to pass in (should be a copy).
        timeout: Maximum execution time in seconds.
        
    Returns:
        dict: {"success": True, "result": value} or {"success": False, "error_type": str, "error_msg": str, "code": code}
    """
    # Defense-in-depth: reject code containing known sandbox escape patterns.
    # The restricted builtins are the primary barrier; this is the secondary check.
    _BLOCKED_PATTERNS = [
        "__class__", "__bases__", "__subclasses__", "__globals__", "__code__",
        "__import__", "breakpoint", "compile(", "exec(", "eval(",
        "getattr", "setattr", "delattr", "globals(", "locals(",
        "open(", "os.", "sys.", "subprocess", "signal",
    ]
    code_lower = code.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            return {
                "success": False,
                "error_type": "SecurityError",
                "error_msg": f"Code contains blocked pattern: '{pattern}'",
                "code": code,
            }

    # Safe builtins for pandas manipulation
    safe_builtins = {
        "len": len, "sum": sum, "min": min, "max": max, "round": round, "abs": abs,
        "sorted": sorted, "list": list, "dict": dict, "str": str, "int": int,
        "float": float, "bool": bool, "True": True, "False": False, "None": None,
        "isinstance": isinstance, "range": range, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "print": lambda *args, **kwargs: None,
    }
    
    # Restricted execution environment
    env = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "__builtins__": safe_builtins,
    }

    result_container = {}
    exception_container = {}

    def _worker():
        try:
            # Try evaluating as a single expression first
            try:
                # Compile to catch syntax errors immediately
                compiled_expr = compile(code, "<string>", "eval")
                res = eval(compiled_expr, env)
                result_container["value"] = res
            except SyntaxError:
                # If it's a statement (e.g. assignment), use exec
                compiled_stmt = compile(code, "<string>", "exec")
                exec(compiled_stmt, env)
                # Look for a variable named 'result', otherwise try to extract a modified df
                if "result" in env:
                    result_container["value"] = env["result"]
                else:
                    result_container["value"] = env.get("df")
        except Exception as e:
            exception_container["exc"] = e

    # Run in a thread for timeout control
    thread = threading.Thread(target=_worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return {
            "success": False,
            "error_type": "TimeoutError",
            "error_msg": f"Code execution exceeded {timeout} seconds.",
            "code": code
        }

    if "exc" in exception_container:
        e = exception_container["exc"]
        return {
            "success": False,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "code": code
        }

    # Format result and enforce limits
    val = result_container.get("value")
    
    if isinstance(val, pd.DataFrame):
        # Cap large dataframes
        if len(val) > DYNAMIC_RESULT_MAX_ROWS:
            val = val.head(DYNAMIC_RESULT_MAX_ROWS)
            
    elif isinstance(val, pd.Series):
        # Convert series to dataframe or scalar depending on shape
        if len(val) == 1:
            val = val.iloc[0]
        else:
            val = val.to_frame()
            if len(val) > DYNAMIC_RESULT_MAX_ROWS:
                val = val.head(DYNAMIC_RESULT_MAX_ROWS)

    # Convert numpy types to native python types for JSON serialization later
    if isinstance(val, (np.integer, np.floating)):
        val = val.item()

    return {"success": True, "result": val}


def run_dynamic_query(
    question: str, df: pd.DataFrame, schema: DatasetSchema, max_retries: int = DYNAMIC_CODE_MAX_RETRIES, value_index: dict | None = None
) -> QueryResult:
    """Attempt dynamic code generation with self-correction retries."""
    import gemini_helper  # Import here to avoid circular dependencies

    # Build a compact schema dict for the prompt
    df_schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "sample_rows": df.head(3).to_dict(orient="records"),
    }

    t0 = time.time()
    code = None
    try:
        code = gemini_helper.generate_pandas_code(question, df_schema, value_index=value_index)
    except Exception as e:
        logger.error("Dynamic code generation failed initially: %s", e)
        return QueryResult(
            operation="dynamic_query", success=False, 
            error="Failed to generate query code using AI.",
            execution_path="dynamic",
        )

    attempts = 0
    while attempts <= max_retries:
        logger.info("Dynamic execution attempt %d/%d with code: %s", attempts + 1, max_retries + 1, code)
        # Always pass a copy of the dataframe
        exec_res = execute_sandboxed(code, df.copy(), timeout=SANDBOX_TIMEOUT_SECONDS)
        
        if exec_res["success"]:
            val = exec_res["result"]
            
            # Determine if result is table or scalar
            is_table = isinstance(val, pd.DataFrame)
            is_empty = (is_table and val.empty) or (not is_table and pd.isna(val))
            
            # For empty/NaN results, still return success, just empty.
            
            return QueryResult(
                operation="dynamic_query",
                success=True,
                table_result=val if is_table else None,
                scalar_result=None if is_table else val,
                explanation="Executed dynamically generated Pandas code.",
                execution_path="dynamic",
                dynamic_code=code,
                dynamic_retry_count=attempts,
                execution_time_ms=round((time.time() - t0) * 1000, 2),
            )
        
        # Handle failure
        error_msg = f"{exec_res['error_type']}: {exec_res['error_msg']}"
        logger.warning("Dynamic execution failed on attempt %d: %s", attempts + 1, error_msg)
        
        if attempts < max_retries:
            try:
                code = gemini_helper.generate_corrected_code(question, code, error_msg, df_schema)
            except Exception as e:
                logger.error("Dynamic code correction failed: %s", e)
                break
        
        attempts += 1

    # If all attempts failed
    return QueryResult(
        operation="dynamic_query",
        success=False,
        error="Could not automatically generate a correct query after multiple attempts. Try simplifying your question.",
        execution_path="dynamic",
        dynamic_retry_count=attempts - 1, # -1 because we incremented at the end of the loop
        execution_time_ms=round((time.time() - t0) * 1000, 2),
    )
