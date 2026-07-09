"""
test_sandbox.py
---------------
Tests for dynamic code execution and sandbox.
"""
from __future__ import annotations
from models import RootIntentModel, SingleIntentModel, ConditionModel
import pandas as pd
import pytest
import math
from unittest.mock import patch, MagicMock
from sandbox import execute_sandboxed, run_dynamic_query
from utils import DatasetSchema, detect_schema

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({'Customer Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'], 'Budget (INR)': [5000000, 9000000, 12000000, 7500000, None], 'Preferred Location': ['Pune', 'Pune', 'Mumbai', 'Pune', 'Mumbai'], 'Property Type': ['2 BHK', '3 BHK', '2 BHK', '3 BHK', '1 BHK'], 'Call Status': ['Interested', 'Not Interested', 'Interested', 'Follow Up', 'Interested'], 'Contact Number': ['9876543210', '9876543211', None, '9876543213', '9876543214'], 'Date Connected': pd.to_datetime(['2026-01-15', '2026-02-20', '2026-03-10', '2026-06-05', '2026-07-01'])})

@pytest.fixture
def schema(sample_df: pd.DataFrame) -> DatasetSchema:
    return detect_schema(sample_df)

class TestExecuteSandboxed:

    def test_valid_aggregation(self, sample_df):
        code = "result = df['Budget (INR)'].mean()"
        res = execute_sandboxed(code, sample_df)
        assert res['success'] is True
        assert res['result'] == sample_df['Budget (INR)'].mean()

    def test_valid_filter(self, sample_df):
        code = "df[df['Preferred Location'] == 'Pune']"
        res = execute_sandboxed(code, sample_df)
        assert res['success'] is True
        assert len(res['result']) == 3

    def test_missing_column(self, sample_df):
        code = "df['Nonexistent'].sum()"
        res = execute_sandboxed(code, sample_df)
        assert res['success'] is False
        assert res['error_type'] == 'KeyError'

    def test_sandbox_escape_import(self, sample_df):
        code = "__import__('os').system('echo hacked')"
        res = execute_sandboxed(code, sample_df)
        assert res['success'] is False
        assert res['error_type'] == 'SecurityError'
        assert '__import__' in res['error_msg']

    def test_sandbox_escape_builtins(self, sample_df):
        code = '().__class__.__bases__[0].__subclasses__()'
        res = execute_sandboxed(code, sample_df)
        assert res['success'] is False
        assert res['error_type'] == 'SecurityError'
        assert '__class__' in res['error_msg']

    def test_sandbox_escape_getattr(self, sample_df):
        code = "getattr(__builtins__, '__import__')('os')"
        res = execute_sandboxed(code, sample_df)
        assert res['success'] is False
        assert res['error_type'] == 'SecurityError'
        assert '__import__' in res['error_msg']

    def test_sandbox_escape_eval(self, sample_df):
        code = "eval('1+1')"
        res = execute_sandboxed(code, sample_df)
        assert res['success'] is False
        assert res['error_type'] == 'SecurityError'
        assert 'eval(' in res['error_msg']

    def test_timeout(self, sample_df):
        code = 'while True: pass'
        res = execute_sandboxed(code, sample_df, timeout=1)
        assert res['success'] is False
        assert res['error_type'] == 'TimeoutError'

    def test_result_capping(self):
        large_df = pd.DataFrame({'A': range(100)})
        code = 'df'
        res = execute_sandboxed(code, large_df)
        assert res['success'] is True
        assert len(res['result']) == 50

    def test_df_not_mutated(self, sample_df):
        code = "df['NewCol'] = 1; result = df"
        copy_df = sample_df.copy()
        res = execute_sandboxed(code, copy_df)
        assert res['success'] is True
        assert 'NewCol' not in copy_df.columns
        assert 'NewCol' in res['result'].columns

    def test_nan_result(self, sample_df):
        code = "result = float('nan')"
        res = execute_sandboxed(code, sample_df)
        assert res['success'] is True
        assert math.isnan(res['result'])

class TestRunDynamicQuery:

    @patch('gemini_helper.generate_pandas_code')
    def test_success_first_attempt(self, mock_gen, sample_df, schema):
        mock_gen.return_value = "df['Budget (INR)'].sum()"
        res = run_dynamic_query('what is the total budget', sample_df, schema)
        assert res.success is True
        assert res.scalar_result == sample_df['Budget (INR)'].sum()
        assert res.dynamic_retry_count == 0

    @patch('gemini_helper.generate_corrected_code')
    @patch('gemini_helper.generate_pandas_code')
    def test_success_after_retry(self, mock_gen, mock_corr, sample_df, schema):
        mock_gen.return_value = "df['BadCol'].sum()"
        mock_corr.return_value = "df['Budget (INR)'].sum()"
        res = run_dynamic_query('what is the total budget', sample_df, schema)
        assert res.success is True
        assert res.scalar_result == sample_df['Budget (INR)'].sum()
        assert res.dynamic_retry_count == 1

    @patch('gemini_helper.generate_corrected_code')
    @patch('gemini_helper.generate_pandas_code')
    def test_all_retries_exhausted(self, mock_gen, mock_corr, sample_df, schema):
        mock_gen.return_value = "df['BadCol'].sum()"
        mock_corr.return_value = "df['BadCol2'].sum()"
        res = run_dynamic_query('what is the total budget', sample_df, schema, max_retries=1)
        assert res.success is False
        assert 'automatically generate a correct query' in res.error
        assert res.dynamic_retry_count == 1

    @patch('gemini_helper.generate_pandas_code')
    def test_gemini_unavailable(self, mock_gen, sample_df, schema):
        import gemini_helper
        mock_gen.side_effect = gemini_helper.GeminiUnavailableError('API Key missing')
        res = run_dynamic_query('what is the total budget', sample_df, schema)
        assert res.success is False
        assert res.execution_path == 'dynamic'
        assert 'Failed to generate query code' in res.error

class TestDynamicIntegration:

    @patch('gemini_helper.generate_pandas_code')
    def test_compound_filter_groupby(self, mock_gen, sample_df, schema):
        mock_gen.return_value = "df[df['Property Type'] == '2 BHK'].groupby('Preferred Location')['Budget (INR)'].mean().reset_index()"
        res = run_dynamic_query('Average budget of 2BHK buyers in Pune', sample_df, schema)
        assert res.success is True
        assert res.table_result is not None
        assert 'Preferred Location' in res.table_result.columns

    @patch('gemini_helper.generate_pandas_code')
    def test_multi_condition_query(self, mock_gen, sample_df, schema):
        mock_gen.return_value = "df[(df['Budget (INR)'] > 8000000) & (df['Call Status'] == 'Interested')]"
        res = run_dynamic_query('Who is interested and has budget > 80L?', sample_df, schema)
        assert res.success is True
        assert len(res.table_result) == 1
        assert res.table_result.iloc[0]['Customer Name'] == 'Charlie'

    @patch('gemini_helper.generate_pandas_code')
    def test_empty_result_handling(self, mock_gen, sample_df, schema):
        mock_gen.return_value = "df[df['Customer Name'] == 'Nobody']"
        res = run_dynamic_query('show nobody', sample_df, schema)
        assert res.success is True
        assert len(res.table_result) == 0

    @patch('gemini_helper.generate_pandas_code')
    def test_markdown_fence_stripping(self, mock_gen, sample_df, schema):
        mock_gen.return_value = '```python\ndf\n```'
        res = run_dynamic_query('test', sample_df, schema, max_retries=0)
        assert res.success is False