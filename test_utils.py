"""
test_utils.py
-------------
Tests for utils.py (loading datasets, schema detection helpers).
"""
import pytest
import pandas as pd
import io
from utils import load_dataframe

class TestLoadDataframe:

    def test_load_csv_success(self):
        csv_data = b"Name, Age, City\nAlice, 30, Pune\nBob, 40, Mumbai"
        f = io.BytesIO(csv_data)
        f.name = "test.csv"
        df = load_dataframe(f)
        assert len(df) == 2
        assert list(df.columns) == ["Name", "Age", "City"]

    def test_duplicate_columns_deduplication(self):
        csv_data = b"Name, Status, Status\nAlice, Active, VIP\nBob, Inactive, Normal"
        f = io.BytesIO(csv_data)
        f.name = "test.csv"
        df = load_dataframe(f)
        assert len(df) == 2
        assert list(df.columns) == ["Name", "Status_1", "Status_2"]
        assert df["Status_1"].iloc[0].strip() == "Active"
        assert df["Status_2"].iloc[0].strip() == "VIP"

    def test_empty_file(self):
        f = io.BytesIO(b"")
        f.name = "empty.csv"
        with pytest.raises(ValueError, match="Could not read the file: No columns to parse"):
            load_dataframe(f)

    def test_only_headers_file(self):
        f = io.BytesIO(b"Name, Age, City")
        f.name = "headers.csv"
        with pytest.raises(ValueError, match="no data rows"):
            load_dataframe(f)
