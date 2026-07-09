import pytest
import pandas as pd
from sandbox import execute_sandboxed

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })

def test_safe_execution(sample_df):
    code = "result = df[df[\"A\"] > 1]"
    res = execute_sandboxed(code, sample_df, timeout=5)
    assert res["success"] is True
    assert len(res["result"]) == 2

def test_timeout_execution(sample_df):
    code = "while True: pass"
    res = execute_sandboxed(code, sample_df, timeout=1)
    assert res["success"] is False
    assert res["error_type"] == "TimeoutError"

def test_ast_blocks_imports(sample_df):
    code = "import os; os.system(\"echo 1\")"
    res = execute_sandboxed(code, sample_df, timeout=5)
    assert res["success"] is False
    assert res["error_type"] == "SecurityError"
    assert "Imports are blocked" in res["error_msg"]

def test_ast_blocks_eval(sample_df):
    code = "eval(\"1 + 1\")"
    res = execute_sandboxed(code, sample_df, timeout=5)
    assert res["success"] is False
    assert res["error_type"] == "SecurityError"
    assert "Function 'eval' is blocked" in res["error_msg"]

def test_ast_blocks_df_query(sample_df):
    code = "df.query(\"A > 1\")"
    res = execute_sandboxed(code, sample_df, timeout=5)
    assert res["success"] is False
    assert res["error_type"] == "SecurityError"
    assert "Method 'query' is blocked" in res["error_msg"]

def test_ast_blocks_dunder(sample_df):
    code = "x = df.__class__"
    res = execute_sandboxed(code, sample_df, timeout=5)
    assert res["success"] is False
    assert res["error_type"] == "SecurityError"
    assert "dunder attribute" in res["error_msg"]
