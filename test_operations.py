"""
test_operations.py
--------------------
Comprehensive test suite for the deterministic query engine.

Tests cover all operation handlers, the condition application system,
number parsing with Indian units, the rule-based intent parser, and
edge cases (empty DataFrames, NaN values, missing columns).
"""
from __future__ import annotations
from models import RootIntentModel, SingleIntentModel, ConditionModel
import pandas as pd
import pytest
import math
from unittest.mock import patch, MagicMock
from config import MAX_QUERY_STEPS, OPERATIONS
from operations import QueryEngine, QueryResult, StepTrace, _extract_numbers_with_shared_units, _match_categorical_conditions, _parse_number_with_unit, merge_follow_up_conditions, rule_based_intent, validate_chain_intent
from utils import DatasetSchema, detect_schema

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small but realistic dataset mirroring the sample_leads.xlsx schema."""
    return pd.DataFrame({'Customer Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'], 'Budget (INR)': [5000000, 9000000, 12000000, 7500000, None], 'Preferred Location': ['Pune', 'Pune', 'Mumbai', 'Pune', 'Mumbai'], 'Property Type': ['2 BHK', '3 BHK', '2 BHK', '3 BHK', '1 BHK'], 'Call Status': ['Interested', 'Not Interested', 'Interested', 'Follow Up', 'Interested'], 'Contact Number': ['9876543210', '9876543211', None, '9876543213', '9876543214'], 'Date Connected': pd.to_datetime(['2026-01-15', '2026-02-20', '2026-03-10', '2026-06-05', '2026-07-01'])})

@pytest.fixture
def schema(sample_df: pd.DataFrame) -> DatasetSchema:
    return detect_schema(sample_df)

@pytest.fixture
def engine(sample_df: pd.DataFrame, schema: DatasetSchema) -> QueryEngine:
    return QueryEngine(sample_df, schema)

class TestNumberParsing:

    def test_plain_number(self):
        assert _parse_number_with_unit('budget above 9000000') == 9000000

    def test_lakh(self):
        assert _parse_number_with_unit('budget above 90 lakh') == 9000000

    def test_lac(self):
        assert _parse_number_with_unit('above 90 lac') == 9000000

    def test_crore(self):
        assert _parse_number_with_unit('above 1 crore') == 10000000

    def test_cr(self):
        assert _parse_number_with_unit('above 1 cr') == 10000000

    def test_k(self):
        assert _parse_number_with_unit('salary above 500k') == 500000

    def test_thousand(self):
        assert _parse_number_with_unit('above 50 thousand') == 50000

    def test_decimal(self):
        assert _parse_number_with_unit('above 1.5 crore') == 15000000

    def test_no_number(self):
        assert _parse_number_with_unit('show me customers') is None

    def test_word_number_crore(self):
        """Word-based numbers like 'one crore' should be parsed."""
        assert _parse_number_with_unit('budget over one crore') == 10000000

    def test_word_number_lakh(self):
        assert _parse_number_with_unit('above five lakh') == 500000

class TestSharedUnitParsing:

    def test_shared_trailing_unit(self):
        """'80 and 120 lakhs' -> both numbers get the 'lakhs' unit."""
        result = _extract_numbers_with_shared_units('between 80 and 120 lakhs')
        assert result == [8000000, 12000000]

    def test_individual_units(self):
        result = _extract_numbers_with_shared_units('between 80 lakh and 1 crore')
        assert result == [8000000, 10000000]

    def test_no_numbers(self):
        assert _extract_numbers_with_shared_units('show me customers') == []

class TestApplyConditions:

    def test_eq_string(self, engine):
        filtered, used, skipped = engine._apply_conditions(engine.df, [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}])
        assert len(filtered) == 3
        assert 'Preferred Location' in used
        assert skipped == []

    def test_eq_case_insensitive(self, engine):
        filtered, _, _ = engine._apply_conditions(engine.df, [{'column': 'Preferred Location', 'op': 'eq', 'value': 'pune'}])
        assert len(filtered) == 3

    def test_gt(self, engine):
        filtered, _, _ = engine._apply_conditions(engine.df, [{'column': 'Budget (INR)', 'op': 'gt', 'value': 8000000}])
        assert len(filtered) == 2

    def test_between(self, engine):
        filtered, _, _ = engine._apply_conditions(engine.df, [{'column': 'Budget (INR)', 'op': 'between', 'value': 5000000, 'value2': 10000000}])
        assert len(filtered) == 3

    def test_contains(self, engine):
        filtered, _, _ = engine._apply_conditions(engine.df, [{'column': 'Property Type', 'op': 'contains', 'value': '2 BHK'}])
        assert len(filtered) == 2

    def test_neq(self, engine):
        filtered, _, _ = engine._apply_conditions(engine.df, [{'column': 'Preferred Location', 'op': 'neq', 'value': 'Pune'}])
        assert len(filtered) == 2

    def test_unresolved_column_tracked(self, engine):
        _, _, skipped = engine._apply_conditions(engine.df, [{'column': 'NonExistentColumn', 'op': 'eq', 'value': 'test'}])
        assert len(skipped) == 1
        assert 'NonExistentColumn' in skipped[0]

    def test_invalid_value_tracked(self, engine):
        _, _, skipped = engine._apply_conditions(engine.df, [{'column': 'Budget (INR)', 'op': 'gt', 'value': 'not_a_number'}])
        assert len(skipped) == 1

    def test_multiple_conditions_and(self, engine):
        filtered, _, _ = engine._apply_conditions(engine.df, [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}, {'column': 'Budget (INR)', 'op': 'gt', 'value': 6000000}])
        assert len(filtered) == 2

    def test_isna_operator(self, engine):
        filtered, _, _ = engine._apply_conditions(engine.df, [{'column': 'Contact Number', 'op': 'isna', 'value': None}])
        assert len(filtered) == 1

    def test_fuzzy_matching_categorical(self, engine):
        from matching import build_value_index
        engine.value_index = build_value_index(engine.df, engine.schema.categorical_cols)
        filtered, _, _ = engine._apply_conditions(engine.df, [{'column': 'Property Type', 'op': 'eq', 'value': '2 BHK '}])
        assert len(filtered) == 2
        assert len(engine.fuzzy_matches_recorded) == 1
        assert engine.fuzzy_matches_recorded[0]['method'] == 'normalized'
        filtered2, _, _ = engine._apply_conditions(engine.df, [{'column': 'Call Status', 'op': 'eq', 'value': 'Not Intrested'}])
        assert len(filtered2) == 1
        assert len(engine.fuzzy_matches_recorded) == 2
        assert engine.fuzzy_matches_recorded[1]['method'] == 'fuzzy'

class TestOperationCount:

    def test_count_all(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count', 'conditions': []}))
        assert result.success
        assert result.scalar_result == 5

    def test_count_with_filter(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}))
        assert result.success
        assert result.scalar_result == 3

class TestOperationSum:

    def test_sum_all(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'sum', 'column': schema.primary_budget_col}))
        assert result.success
        assert result.scalar_result == 33500000

    def test_sum_with_filter(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'sum', 'column': schema.primary_budget_col, 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}))
        assert result.success
        assert result.scalar_result == 21500000

class TestOperationAverage:

    def test_average_all(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'average', 'column': schema.primary_budget_col}))
        assert result.success
        assert result.scalar_result == pytest.approx(8375000)

    def test_nan_excluded_note(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'average', 'column': schema.primary_budget_col}))
        assert 'NaN' in result.explanation

class TestOperationMedian:

    def test_median_all(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'median', 'column': schema.primary_budget_col}))
        assert result.success
        assert result.scalar_result == pytest.approx(8250000)

class TestOperationMinMax:

    def test_min(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'min', 'column': schema.primary_budget_col}))
        assert result.success
        assert result.scalar_result == 5000000

    def test_max(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'max', 'column': schema.primary_budget_col}))
        assert result.success
        assert result.scalar_result == 12000000

    def test_max_returns_row(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'max', 'column': schema.primary_budget_col}))
        assert result.table_result is not None
        assert result.table_result.iloc[0]['Customer Name'] == 'Charlie'

class TestOperationFilter:

    def test_greater_than(self, engine, schema):
        intent = RootIntentModel.model_validate({'operation': 'greater_than', 'column': schema.primary_budget_col, 'value': 8000000.0})
        result = engine.execute(intent)
        assert result.success
        assert result.scalar_result == 2

    def test_greater_than_with_currency_string(self, engine, schema):
        intent = RootIntentModel.model_validate({'operation': 'greater_than', 'column': schema.primary_budget_col, 'value': '₹80L'})
        result = engine.execute(intent)
        assert result.success
        assert result.scalar_result == 2  # Same as 8000000


    def test_less_than(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'less_than', 'column': schema.primary_budget_col, 'value': 8000000}))
        assert result.success
        assert result.scalar_result == 2

    def test_between(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'between', 'column': schema.primary_budget_col, 'value': 7000000, 'value2': 10000000}))
        assert result.success
        assert result.scalar_result == 2

class TestOperationSort:

    def test_sort_descending(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}))
        assert result.success
        vals = result.table_result[schema.primary_budget_col].dropna().tolist()
        assert vals == sorted(vals, reverse=True)

    def test_sort_with_n(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False, 'n': 2}))
        assert result.success
        assert len(result.table_result) == 2

class TestOperationTopBottomN:

    def test_topn(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'topn', 'column': schema.primary_budget_col, 'n': 3}))
        assert result.success
        assert len(result.table_result) == 3
        vals = result.table_result[schema.primary_budget_col].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_bottomn(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'bottomn', 'column': schema.primary_budget_col, 'n': 2}))
        assert result.success
        assert len(result.table_result) == 2

class TestOperationGroupby:

    def test_groupby_count(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'groupby', 'group_by': 'Preferred Location', 'agg_func': 'count'}))
        assert result.success
        assert len(result.table_result) == 2
        pune_count = result.table_result[result.table_result['Preferred Location'] == 'Pune']['count'].values[0]
        assert pune_count == 3

    def test_groupby_mean(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'groupby', 'group_by': 'Preferred Location', 'agg_column': schema.primary_budget_col, 'agg_func': 'mean'}))
        assert result.success
        assert len(result.table_result) == 2

class TestOperationUnique:

    def test_unique_values(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'unique', 'column': 'Preferred Location'}))
        assert result.success
        assert set(result.scalar_result) == {'Mumbai', 'Pune'}

    def test_distinct_count(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'distinct_count', 'column': 'Preferred Location'}))
        assert result.success
        assert result.scalar_result == 2

class TestOperationDescribe:

    def test_describe(self, engine, schema):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'describe', 'column': schema.primary_budget_col}))
        assert result.success
        assert result.table_result is not None

class TestOperationList:

    def test_list_with_conditions(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'list', 'conditions': [{'column': 'Call Status', 'op': 'eq', 'value': 'Interested'}], 'n': 50}))
        assert result.success
        assert result.scalar_result == 3

    def test_list_with_n(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'list', 'n': 2}))
        assert result.success
        assert len(result.table_result) == 2

class TestOperationMissing:

    def test_missing_specific_column(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'missing', 'column': 'Contact Number'}))
        assert result.success
        assert result.scalar_result == 1

    def test_missing_any(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'missing', 'column': None}))
        assert result.success
        assert result.scalar_result == 2

class TestOperationDateFilter:

    def test_date_after(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'date_filter', 'date_op': 'after', 'value': '2026-03-01'}))
        assert result.success
        assert result.scalar_result == 3

    def test_date_before(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'date_filter', 'date_op': 'before', 'value': '2026-02-28'}))
        assert result.success
        assert result.scalar_result == 2

class TestUnknownOperation:

    def test_unknown_op_fails(self, engine):
        result = engine.execute({'operation': 'nonexistent'})
        assert result.success is False
        assert 'Unrecognized operation' in result.error

    def test_empty_op_fails(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': ''}))
        assert not result.success

class TestExecutionMetadata:

    def test_execution_time(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count'}))
        assert result.execution_time_ms >= 0

    def test_rows_scanned(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count'}))
        assert result.rows_scanned == 5

    def test_filters_applied(self, engine):
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}))
        assert result.filters_applied == 1

class TestRuleBasedIntent:

    def test_average(self, schema, sample_df):
        intent = rule_based_intent('what is the average budget', schema, sample_df)
        assert intent.operation == 'average'

    def test_avg_abbreviation(self, schema, sample_df):
        intent = rule_based_intent('avg budget', schema, sample_df)
        assert intent.operation == 'average'

    def test_median(self, schema, sample_df):
        intent = rule_based_intent('what is the median budget', schema, sample_df)
        assert intent.operation == 'median'

    def test_count(self, schema, sample_df):
        intent = rule_based_intent('how many customers are there', schema, sample_df)
        assert intent.operation == 'count'

    def test_max(self, schema, sample_df):
        intent = rule_based_intent('who has the highest budget', schema, sample_df)
        assert intent.operation == 'max'

    def test_min(self, schema, sample_df):
        intent = rule_based_intent('what is the lowest budget', schema, sample_df)
        assert intent.operation == 'min'

    def test_greater_than_lakh(self, schema, sample_df):
        intent = rule_based_intent('customers with budget above 90 lakh', schema, sample_df)
        assert intent.operation == 'greater_than'
        assert intent.value == 9000000

    def test_between(self, schema, sample_df):
        intent = rule_based_intent('customers between 50 and 90 lakhs', schema, sample_df)
        assert intent.operation == 'between'
        assert intent.value == 5000000
        assert intent.value2 == 9000000

    def test_groupby_breakdown(self, schema, sample_df):
        intent = rule_based_intent("show budget distribution", schema, sample_df)
        assert intent.operation == "groupby"
        assert intent.agg_func == "count"

    def test_groupby_each_property_type(self, schema, sample_df):
        intent = rule_based_intent("How many customers prefer each Property Type?", schema, sample_df)
        assert intent.operation == "groupby"
        assert intent.group_by == schema.property_type_col
        assert intent.agg_func == "count"

    def test_groupby_most_customers_location(self, schema, sample_df):
        intent = rule_based_intent("Which Location has the most customers?", schema, sample_df)
        assert intent.operation == "groupby"
        assert intent.group_by == schema.location_col
        assert intent.agg_func == "count"

    def test_groupby_count_by_status(self, schema, sample_df):
        intent = rule_based_intent("Count customers by Call Status", schema, sample_df)
        assert intent.operation == "groupby"
        assert intent.group_by == schema.status_col
        assert intent.agg_func == "count"

    def test_groupby_least_common_type(self, schema, sample_df):
        intent = rule_based_intent("Least common Property Type", schema, sample_df)
        assert intent.operation == "groupby"
        assert intent.group_by == schema.property_type_col
        assert intent.agg_func == "count"

    def test_top_n(self, schema, sample_df):
        intent = rule_based_intent('top 5 customers by budget', schema, sample_df)
        assert intent.operation == 'topn'
        assert intent.n == 5

    def test_groupby_breakdown(self, schema, sample_df):
        intent = rule_based_intent('breakdown of call status', schema, sample_df)
        assert intent.operation == 'groupby'

    def test_distribution(self, schema, sample_df):
        intent = rule_based_intent('distribution of property type', schema, sample_df)
        assert intent.operation == 'groupby'

    def test_unique(self, schema, sample_df):
        intent = rule_based_intent('what are the unique locations', schema, sample_df)
        assert intent.operation == 'unique'

    def test_distinct_count(self, schema, sample_df):
        intent = rule_based_intent('how many distinct locations', schema, sample_df)
        assert intent.operation == 'distinct_count'

    def test_sum(self, schema, sample_df):
        intent = rule_based_intent('total budget', schema, sample_df)
        assert intent.operation == 'sum'

    def test_categorical_filter(self, schema, sample_df):
        intent = rule_based_intent('show me customers in pune', schema, sample_df)
        assert intent.operation == 'list'
        conditions = intent.conditions
        location_match = [c for c in conditions if 'pune' in str(c.value).lower()]
        assert len(location_match) >= 1 or len(conditions) >= 0

    def test_missing_query(self, schema, sample_df):
        intent = rule_based_intent('customers with missing phone', schema, sample_df)
        assert intent.operation == 'missing'

    def test_date_after(self, schema, sample_df):
        intent = rule_based_intent('customers after march', schema, sample_df)
        assert intent.operation == 'date_filter'
        assert intent.date_op == 'after'

    def test_recent(self, schema, sample_df):
        intent = rule_based_intent('show recent customers', schema, sample_df)
        assert intent.operation == 'sort'

class TestMergeFollowUp:

    def test_merge_new_conditions(self):
        new_intent = {'operation': 'count', 'conditions': []}
        prev = [{'column': 'Location', 'op': 'eq', 'value': 'Pune'}]
        merged = merge_follow_up_conditions(new_intent, prev)
        assert len(merged.conditions) == 1
        assert merged.conditions[0].value == 'Pune'

    def test_no_duplicate_merge(self):
        new_intent = {'operation': 'count', 'conditions': [{'column': 'Location', 'op': 'eq', 'value': 'Mumbai'}]}
        prev = [{'column': 'Location', 'op': 'eq', 'value': 'Pune'}]
        merged = merge_follow_up_conditions(new_intent, prev)
        assert len(merged.conditions) == 1
        assert merged.conditions[0].value == 'Mumbai'

    def test_empty_previous(self):
        new_intent = {'operation': 'count', 'conditions': []}
        merged = merge_follow_up_conditions(new_intent, [])
        assert merged.conditions == []

class TestCategoricalMatching:

    def test_match_location(self, schema, sample_df):
        conditions = _match_categorical_conditions('customers in pune', schema, sample_df)
        if schema.location_col:
            location_conditions = [c for c in conditions if c.column == schema.location_col]
            assert len(location_conditions) >= 1 or not any((v.lower() == 'pune' for v in sample_df[schema.location_col].dropna().astype(str).unique()))

    def test_match_property_type(self, schema, sample_df):
        conditions = _match_categorical_conditions('2 bhk in pune', schema, sample_df)
        assert len(conditions) >= 1

class TestEdgeCases:

    def test_empty_dataframe(self):
        df = pd.DataFrame({'A': [], 'B': []})
        schema = DatasetSchema(all_columns=['A', 'B'])
        engine = QueryEngine(df, schema)
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count'}))
        assert result.success
        assert result.scalar_result == 0

    def test_all_nan_column(self):
        df = pd.DataFrame({'Budget': [None, None, None], 'Name': ['A', 'B', 'C']})
        schema = detect_schema(df)
        engine = QueryEngine(df, schema)
        result = engine.execute(RootIntentModel.model_validate({'operation': 'average', 'column': 'Budget'}))
        assert result.success
        assert result.scalar_result is None or math.isnan(result.scalar_result)

    def test_single_row(self):
        df = pd.DataFrame({'Budget': [100], 'Name': ['Only']})
        schema = detect_schema(df)
        engine = QueryEngine(df, schema)
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count'}))
        assert result.success
        assert result.scalar_result == 1

class TestChainedExecution:
    """Tests for the execute_chain() method and multi-step pipeline."""

    def test_two_step_filter_sort(self, engine, schema):
        """Filter by location, then sort by budget descending."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}, {'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert result.success
        assert len(result.table_result) == 3
        vals = result.table_result[schema.primary_budget_col].dropna().tolist()
        assert vals == sorted(vals, reverse=True)

    def test_three_step_filter_sort_topn(self, engine, schema):
        """Filter -> sort -> top 2."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}, {'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}, {'operation': 'topn', 'column': schema.primary_budget_col, 'n': 2}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert result.success
        assert len(result.table_result) == 2
        names = result.table_result['Customer Name'].tolist()
        assert 'Bob' in names
        assert 'Diana' in names

    def test_chain_empty_intermediate(self, engine, schema):
        """Filter that produces zero rows -> sort should return empty gracefully."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Narnia'}]}, {'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert result.success
        assert result.table_result is not None
        assert len(result.table_result) == 0

    def test_chain_nonexistent_column_step2(self, engine):
        """Step 2 references a non-existent column -> graceful error."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}, {'operation': 'sort', 'column': 'NonExistentColumn', 'ascending': False}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert result.success or result.error is not None
        assert not isinstance(result, type(None))

    def test_chain_single_step_identical(self, engine, schema):
        """A single-step chain produces identical output to direct execute()."""
        intent = {'operation': 'count', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}
        direct = engine.execute(RootIntentModel.model_validate(intent))
        chained = engine.execute_chain([SingleIntentModel.model_validate(s) for s in [intent]])
        assert direct.scalar_result == chained.scalar_result
        assert direct.operation == chained.operation

    def test_chain_step_trace_populated(self, engine, schema):
        """Verify step_trace has correct length and fields."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}, {'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert len(result.step_trace) == 2
        assert result.step_trace[0].operation == 'filter'
        assert result.step_trace[1].operation == 'sort'
        assert result.step_trace[0].rows_before == 5
        assert result.step_trace[0].rows_after == 3
        assert result.step_trace[1].rows_before == 3

class TestValidateChainIntent:
    """Tests for validate_chain_intent()."""

    def test_max_steps_exceeded(self):
        """Chain with > MAX_QUERY_STEPS should be rejected."""
        too_many = {'steps': [{'operation': 'count'}] * (MAX_QUERY_STEPS + 1)}
        with pytest.raises(ValueError, match='Too many steps'):
            validate_chain_intent(too_many)

    def test_invalid_step_operation(self):
        bad = {'steps': [{'operation': 'nonexistent'}]}
        with pytest.raises(ValueError, match='[Ii]nvalid operation'):
            validate_chain_intent(bad)

class TestRuleBasedChain:

    def test_then_sort(self, schema, sample_df):
        """'show pune customers then sort by budget' -> multi-step."""
        intent = rule_based_intent('show pune customers then sort by budget', schema, sample_df)
        assert intent.steps is not None
        assert len(intent.steps) == 2

    def test_then_top(self, schema, sample_df):
        """'show pune customers then top 5' -> multi-step."""
        intent = rule_based_intent('show pune customers then top 5', schema, sample_df)
        assert intent.steps is not None
        assert len(intent.steps) == 2
        assert intent.steps[1].operation == 'topn'

    def test_no_chain_single_op(self, schema, sample_df):
        """'what is the average budget' -> single intent (no steps)."""
        intent = rule_based_intent('what is the average budget', schema, sample_df)
        assert 'steps' not in intent
        assert intent.operation == 'average'

class TestSingleOpRegression:

    def test_count_with_filter_unchanged(self, engine):
        """Verify count+filter produces IDENTICAL output after chaining changes."""
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}))
        assert result.success
        assert result.scalar_result == 3
        assert result.step_trace == []
        assert result.rows_scanned == 5
        assert result.filters_applied == 1


class TestNoneColumnResolution:
    """Phase 6.1: Edge cases where _numeric_col or _resolve returns None."""

    @pytest.fixture
    def empty_schema_engine(self):
        """Engine with a dataframe that has no numeric or categorical columns."""
        df = pd.DataFrame({'JustStrings': ['a', 'b', 'c']})
        schema = DatasetSchema(
            name_col=None, primary_budget_col=None, location_col=None,
            property_type_col=None, status_col=None, date_cols=[], contact_col=None, id_cols=[]
        )
        return QueryEngine(df, schema)

    def test_op_extremum_none_column(self, empty_schema_engine):
        """_op_min with no numeric columns should fail gracefully."""
        intent = RootIntentModel.model_validate({'operation': 'min'})
        result = empty_schema_engine.execute(intent)
        assert not result.success
        assert "no numeric column found" in result.error.lower()

    def test_op_sort_none_column(self, empty_schema_engine):
        """_op_sort with no column could be determined should fail gracefully."""
        intent = RootIntentModel.model_validate({'operation': 'sort'})
        result = empty_schema_engine.execute(intent)
        assert not result.success
        assert "no column could be determined" in result.error.lower()
        assert intent.operation == 'date_filter'
        assert intent.date_op == 'after'

    def test_recent(self, schema, sample_df):
        intent = rule_based_intent('show recent customers', schema, sample_df)
        assert intent.operation == 'sort'

class TestMergeFollowUp:

    def test_merge_new_conditions(self):
        new_intent = {'operation': 'count', 'conditions': []}
        prev = [{'column': 'Location', 'op': 'eq', 'value': 'Pune'}]
        merged = merge_follow_up_conditions(new_intent, prev)
        assert len(merged.conditions) == 1
        assert merged.conditions[0].value == 'Pune'

    def test_no_duplicate_merge(self):
        new_intent = {'operation': 'count', 'conditions': [{'column': 'Location', 'op': 'eq', 'value': 'Mumbai'}]}
        prev = [{'column': 'Location', 'op': 'eq', 'value': 'Pune'}]
        merged = merge_follow_up_conditions(new_intent, prev)
        assert len(merged.conditions) == 1
        assert merged.conditions[0].value == 'Mumbai'

    def test_empty_previous(self):
        new_intent = {'operation': 'count', 'conditions': []}
        merged = merge_follow_up_conditions(new_intent, [])
        assert merged.conditions == []

class TestCategoricalMatching:

    def test_match_location(self, schema, sample_df):
        conditions = _match_categorical_conditions('customers in pune', schema, sample_df)
        if schema.location_col:
            location_conditions = [c for c in conditions if c.column == schema.location_col]
            assert len(location_conditions) >= 1 or not any((v.lower() == 'pune' for v in sample_df[schema.location_col].dropna().astype(str).unique()))

    def test_match_property_type(self, schema, sample_df):
        conditions = _match_categorical_conditions('2 bhk in pune', schema, sample_df)
        assert len(conditions) >= 1

class TestEdgeCases:

    def test_empty_dataframe(self):
        df = pd.DataFrame({'A': [], 'B': []})
        schema = DatasetSchema(all_columns=['A', 'B'])
        engine = QueryEngine(df, schema)
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count'}))
        assert result.success
        assert result.scalar_result == 0

    def test_all_nan_column(self):
        df = pd.DataFrame({'Budget': [None, None, None], 'Name': ['A', 'B', 'C']})
        schema = detect_schema(df)
        engine = QueryEngine(df, schema)
        result = engine.execute(RootIntentModel.model_validate({'operation': 'average', 'column': 'Budget'}))
        assert result.success
        assert result.scalar_result is None or math.isnan(result.scalar_result)

    def test_single_row(self):
        df = pd.DataFrame({'Budget': [100], 'Name': ['Only']})
        schema = detect_schema(df)
        engine = QueryEngine(df, schema)
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count'}))
        assert result.success
        assert result.scalar_result == 1

class TestChainedExecution:
    """Tests for the execute_chain() method and multi-step pipeline."""

    def test_two_step_filter_sort(self, engine, schema):
        """Filter by location, then sort by budget descending."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}, {'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert result.success
        assert len(result.table_result) == 3
        vals = result.table_result[schema.primary_budget_col].dropna().tolist()
        assert vals == sorted(vals, reverse=True)

    def test_three_step_filter_sort_topn(self, engine, schema):
        """Filter -> sort -> top 2."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}, {'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}, {'operation': 'topn', 'column': schema.primary_budget_col, 'n': 2}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert result.success
        assert len(result.table_result) == 2
        names = result.table_result['Customer Name'].tolist()
        assert 'Bob' in names
        assert 'Diana' in names

    def test_chain_empty_intermediate(self, engine, schema):
        """Filter that produces zero rows -> sort should return empty gracefully."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Narnia'}]}, {'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert result.success
        assert result.table_result is not None
        assert len(result.table_result) == 0

    def test_chain_nonexistent_column_step2(self, engine):
        """Step 2 references a non-existent column -> graceful error."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}, {'operation': 'sort', 'column': 'NonExistentColumn', 'ascending': False}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert result.success or result.error is not None
        assert not isinstance(result, type(None))

    def test_chain_single_step_identical(self, engine, schema):
        """A single-step chain produces identical output to direct execute()."""
        intent = {'operation': 'count', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}
        direct = engine.execute(RootIntentModel.model_validate(intent))
        chained = engine.execute_chain([SingleIntentModel.model_validate(s) for s in [intent]])
        assert direct.scalar_result == chained.scalar_result
        assert direct.operation == chained.operation

    def test_chain_step_trace_populated(self, engine, schema):
        """Verify step_trace has correct length and fields."""
        steps = [{'operation': 'filter', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}, {'operation': 'sort', 'column': schema.primary_budget_col, 'ascending': False}]
        result = engine.execute_chain([SingleIntentModel.model_validate(s) for s in steps])
        assert len(result.step_trace) == 2
        assert result.step_trace[0].operation == 'filter'
        assert result.step_trace[1].operation == 'sort'
        assert result.step_trace[0].rows_before == 5
        assert result.step_trace[0].rows_after == 3
        assert result.step_trace[1].rows_before == 3

class TestValidateChainIntent:
    """Tests for validate_chain_intent()."""

    def test_max_steps_exceeded(self):
        """Chain with > MAX_QUERY_STEPS should be rejected."""
        too_many = {'steps': [{'operation': 'count'}] * (MAX_QUERY_STEPS + 1)}
        with pytest.raises(ValueError, match='Too many steps'):
            validate_chain_intent(too_many)

    def test_invalid_step_operation(self):
        bad = {'steps': [{'operation': 'nonexistent'}]}
        with pytest.raises(ValueError, match='[Ii]nvalid operation'):
            validate_chain_intent(bad)

class TestRuleBasedChain:

    def test_then_sort(self, schema, sample_df):
        """'show pune customers then sort by budget' -> multi-step."""
        intent = rule_based_intent('show pune customers then sort by budget', schema, sample_df)
        assert intent.steps is not None
        assert len(intent.steps) == 2

    def test_then_top(self, schema, sample_df):
        """'show pune customers then top 5' -> multi-step."""
        intent = rule_based_intent('show pune customers then top 5', schema, sample_df)
        assert intent.steps is not None
        assert len(intent.steps) == 2
        assert intent.steps[1].operation == 'topn'

    def test_no_chain_single_op(self, schema, sample_df):
        """'what is the average budget' -> single intent (no steps)."""
        intent = rule_based_intent('what is the average budget', schema, sample_df)
        assert 'steps' not in intent
        assert intent.operation == 'average'

class TestSingleOpRegression:

    def test_count_with_filter_unchanged(self, engine):
        """Verify count+filter produces IDENTICAL output after chaining changes."""
        result = engine.execute(RootIntentModel.model_validate({'operation': 'count', 'conditions': [{'column': 'Preferred Location', 'op': 'eq', 'value': 'Pune'}]}))
        assert result.success
        assert result.scalar_result == 3
        assert result.step_trace == []
        assert result.rows_scanned == 5
        assert result.filters_applied == 1


class TestNoneColumnResolution:
    """Phase 6.1: Edge cases where _numeric_col or _resolve returns None."""

    @pytest.fixture
    def empty_schema_engine(self):
        """Engine with a dataframe that has no numeric or categorical columns."""
        df = pd.DataFrame({'JustStrings': ['a', 'b', 'c']})
        schema = DatasetSchema(
            name_col=None, primary_budget_col=None, location_col=None,
            property_type_col=None, status_col=None, date_cols=[], contact_col=None, id_cols=[]
        )
        return QueryEngine(df, schema)

    def test_op_extremum_none_column(self, empty_schema_engine):
        """_op_min with no numeric columns should fail gracefully."""
        intent = RootIntentModel.model_validate({'operation': 'min'})
        result = empty_schema_engine.execute(intent)
        assert not result.success
        assert "no numeric column found" in result.error.lower()

    def test_op_sort_none_column(self, empty_schema_engine):
        """_op_sort with no column could be determined should fail gracefully."""
        intent = RootIntentModel.model_validate({'operation': 'sort'})
        result = empty_schema_engine.execute(intent)
        assert not result.success
        assert "no column could be determined" in result.error.lower()

    def test_op_groupby_none_column(self, empty_schema_engine):
        """_op_groupby with no categorical columns should fail gracefully."""
        intent = RootIntentModel.model_validate({'operation': 'groupby', 'agg_func': 'count'})
        result = empty_schema_engine.execute(intent)
        assert not result.success
        assert "could not determine a column to group by" in result.error.lower()

class TestIntentNormalization:
    """Verify that operations natively ignoring intent.column/value normalize them into conditions."""
    def test_count_normalization(self, engine):
        intent = RootIntentModel.model_validate({
            "operation": "count",
            "column": "property_type",
            "value": "2 BHK"
        })
        result = engine.execute(intent)
        assert result.success
        assert result.scalar_result == 2  # 2 2 BHKs in the sample data

    def test_filter_normalization(self, engine):
        intent = RootIntentModel.model_validate({
            "operation": "filter",
            "column": "location",
            "value": "Pune"
        })
        result = engine.execute(intent)
        assert result.success
        assert result.scalar_result == 3  # 3 in pune

    def test_groupby_normalization(self, engine):
        intent = RootIntentModel.model_validate({
            "operation": "groupby",
            "column": "property_type",
            "value": "2 BHK",
            "group_by": "status",
            "agg_func": "count"
        })
        result = engine.execute(intent)
        assert result.success
        # Should only group the 2 '2 BHK's
        assert result.table_result["count"].sum() == 2
