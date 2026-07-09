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
    # 1. AST-based Security Validation
    import ast
    
    class SecurityScanner(ast.NodeVisitor):
        def visit_Import(self, node):
            raise ValueError(f"Imports are blocked (found '{node.names[0].name}')")
        
        def visit_ImportFrom(self, node):
            raise ValueError(f"Imports are blocked (found '{node.module}')")
            
        def visit_Call(self, node):
            # block eval, exec, getattr, setattr, globals, locals, open, compile
            if isinstance(node.func, ast.Name):
                blocked = {"eval", "exec", "getattr", "setattr", "delattr", "globals", "locals", "open", "compile", "__import__"}
                if node.func.id in blocked:
                    raise ValueError(f"Function '{node.func.id}' is blocked")
            # block df.query and df.eval
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in {"query", "eval", "applymap", "apply"}:
                    raise ValueError(f"Method '{node.func.attr}' is blocked for security")
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            if node.attr.startswith("__"):
                raise ValueError(f"Access to dunder attribute '{node.attr}' is blocked")
            self.generic_visit(node)
            
    try:
        tree = ast.parse(code)
        SecurityScanner().visit(tree)
    except Exception as e:
        return {
            "success": False,
            "error_type": "SecurityError",
            "error_msg": str(e),
            "code": code
        }

    import multiprocessing
    
    manager = multiprocessing.Manager()
    result_container = manager.dict()
    
    # Process execution
    p = multiprocessing.Process(target=_sandbox_worker, args=(code, df, result_container))
    p.daemon = True
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "success": False,
            "error_type": "TimeoutError",
            "error_msg": f"Code execution exceeded {timeout} seconds.",
            "code": code
        }

    if "exc" in result_container:
        e = result_container["exc"]
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

def _sandbox_worker(code: str, df: pd.DataFrame, result_container: dict):
    """Top-level worker function for multiprocessing picklability."""
    safe_builtins = {
        "len": len, "sum": sum, "min": min, "max": max, "round": round, "abs": abs,
        "sorted": sorted, "list": list, "dict": dict, "str": str, "int": int,
        "float": float, "bool": bool, "True": True, "False": False, "None": None,
        "isinstance": isinstance, "range": range, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "print": lambda *args, **kwargs: None,
    }
    
    env = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "__builtins__": safe_builtins,
    }
    
    try:
        try:
            compiled_expr = compile(code, "<string>", "eval")
            res = eval(compiled_expr, env)
            result_container["value"] = res
        except SyntaxError:
            compiled_stmt = compile(code, "<string>", "exec")
            exec(compiled_stmt, env)
            if "result" in env:
                result_container["value"] = env["result"]
            else:
                result_container["value"] = env.get("df")
    except Exception as e:
        result_container["exc"] = e



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
