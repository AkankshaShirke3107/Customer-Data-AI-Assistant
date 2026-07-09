"""
test_schema_detection.py
------------------------
Unit tests for utils._matches_any and schema detection keyword logic,
ensuring false-positives are caught and proper regex word boundaries are used.
"""

from __future__ import annotations

import pytest

from utils import _matches_any

class TestMatchesAny:
    def test_exact_match(self):
        assert _matches_any("id", ["id"]) is True
        assert _matches_any("name", ["name"]) is True
        assert _matches_any("budget", ["budget"]) is True

    def test_word_boundary_prefix_match(self):
        # 'ref' should match 'Reference' because of prefix word boundary \b
        assert _matches_any("Reference", ["ref"]) is True
        assert _matches_any("Ref #", ["ref"]) is True
        assert _matches_any("Lead Name", ["lead name"]) is True

    def test_case_insensitive(self):
        assert _matches_any("ID", ["id"]) is True
        assert _matches_any("REFERENCE", ["ref"]) is True

    def test_false_positives(self):
        # 'Preferred Location' contains 'ref' inside the word 'preferred', should not match
        assert _matches_any("Preferred Location", ["ref"]) is False
        # 'Valid' contains 'id' inside, should not match
        assert _matches_any("Valid", ["id"]) is False
        # 'Paid Amount' contains 'id' inside 'paid', should not match
        assert _matches_any("Paid Amount", ["id"]) is False
        # 'guid' contains 'id' inside, should not match
        assert _matches_any("guid", ["id"]) is False

    def test_multi_word_column_names(self):
        # Should match 'id'
        assert _matches_any("Valid ID", ["id"]) is True
        assert _matches_any("ID Number", ["id"]) is True
        assert _matches_any("Property Type", ["property type"]) is True
        assert _matches_any("Contact Phone Number", ["phone"]) is True
