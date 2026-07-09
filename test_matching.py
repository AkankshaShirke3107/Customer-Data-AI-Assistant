import pandas as pd
from matching import normalize_value, build_value_index, fuzzy_match_category

def test_normalize_value():
    assert normalize_value("Pune ") == "pune"
    assert normalize_value("PUNE") == "pune"
    assert normalize_value("Pune City") == "pune"
    assert normalize_value("Mumbai,") == "mumbai"
    assert normalize_value("  Delhi  ") == "delhi"
    assert normalize_value(None) == ""
    assert normalize_value(float("nan")) == ""

def test_build_value_index():
    df = pd.DataFrame({
        "Location": ["Pune", "Pune ", "Mumbai", "Bombay", "PUNE", "Pune City", "Mumbai"],
        "Other": [1, 2, 3, 4, 5, 6, 7]
    })
    
    index = build_value_index(df, ["Location"])
    assert "Location" in index
    
    # Pune is the most frequent variant mapping to "pune"
    assert index["Location"]["pune"] == "Pune"
    
    # Mumbai is the most frequent variant mapping to "mumbai"
    assert index["Location"]["mumbai"] == "Mumbai"
    
    # Bombay maps to bombay
    assert index["Location"]["bombay"] == "Bombay"

def test_fuzzy_match_category():
    df = pd.DataFrame({
        "Location": ["Pune", "Mumbai", "Delhi"]
    })
    index = build_value_index(df, ["Location"])
    
    # Exact match
    res1 = fuzzy_match_category("Pune", "Location", index)
    assert res1["method"] == "exact"
    assert res1["matched"] == "Pune"
    assert res1["score"] == 100.0
    
    # Normalized match
    res2 = fuzzy_match_category(" PUNE city", "Location", index)
    assert res2["method"] == "normalized"
    assert res2["matched"] == "Pune"
    assert res2["score"] == 100.0
    
    # Fuzzy match (Long string > 5 chars)
    res3 = fuzzy_match_category("Pomes", "Location", index)  # Not close enough if threshold is 85
    assert res3 is None
    
    # Short string typo that passes dynamic threshold (Delh vs Delhi)
    # Length is 4 vs 5. Ratio is 88.8.
    res4 = fuzzy_match_category("Delh", "Location", index)
    assert res4["method"] == "fuzzy"
    assert res4["matched"] == "Delhi"
    
    # Overmatching prevention: Pune vs Pane
    # Length is 4, ratio is 75.0. Dynamic threshold is 80.0, so this should NOT match.
    # This proves we don't start over-matching unrelated values.
    res_overmatch = fuzzy_match_category("Pane", "Location", index)
    assert res_overmatch is None

    # Typo that passes threshold explicitly
    res5 = fuzzy_match_category("Mumabi", "Location", index, threshold=80.0)
    assert res5["method"] == "fuzzy"
    assert res5["matched"] == "Mumbai"
    
    # Property Type short strings with digit protection
    df_prop = pd.DataFrame({
        "Property": ["1 BHK", "2 BHK", "3 BHK"]
    })
    index_prop = build_value_index(df_prop, ["Property"])
    
    # Short string typo (length 5): 2 bkk should match 2 BHK
    res_prop1 = fuzzy_match_category("2 bkk", "Property", index_prop)
    assert res_prop1 is not None
    assert res_prop1["matched"] == "2 BHK"
    
    # Short string semantic mismatch: 4 bhk should NOT match 2 bhk even though ratio is 80.0
    res_prop2 = fuzzy_match_category("4 bhk", "Property", index_prop)
    assert res_prop2 is None
    
    # Empty/NaN
    assert fuzzy_match_category(None, "Location", index) is None
