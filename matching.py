"""
matching.py
-----------
Deterministic fuzzy and normalized string matching for categorical columns.

Provides a shared normalization layer that both the fixed-operation path and 
the dynamic code-generation path use to gracefully handle inconsistent 
real-world data entry (e.g. "Pune ", "PUNE", "Bombay").
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def normalize_value(value: Any) -> str:
    """Safely normalize a string value for comparison.
    
    Transformations applied:
    - Coerce to string
    - Strip leading/trailing whitespace
    - Lowercase
    - Collapse multiple internal spaces into a single space
    - Remove trivial noise words like "city" at the end (e.g., "Pune City" -> "pune")
    - Remove basic punctuation (commas, periods)
    
    Returns an empty string if the value is NaN, None, or empty.
    """
    if pd.isna(value) or value is None:
        return ""
    
    # Coerce to string
    s = str(value)
    
    # Lowercase and strip whitespace
    s = s.lower().strip()
    
    # Remove basic punctuation
    s = re.sub(r"[,.]", "", s)
    
    # Remove trailing noise words
    s = re.sub(r"\bcity\b$", "", s)
    
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    
    return s.strip()


def build_value_index(df: pd.DataFrame, categorical_columns: list[str]) -> dict[str, dict[str, str]]:
    """Build a fast lookup index mapping normalized values to their canonical originals.
    
    Args:
        df: The loaded dataframe.
        categorical_columns: List of column names to index.
        
    Returns:
        dict: A nested dictionary structured as {column_name: {normalized_value: canonical_value}}
              If multiple distinct original values map to the same normalized value, 
              the most frequent original value is chosen as the canonical one.
    """
    index: dict[str, dict[str, str]] = {}
    
    for col in categorical_columns:
        if col not in df.columns:
            continue
            
        col_index: dict[str, str] = {}
        
        # We drop na and get value counts to break ties (pick most common variant)
        val_counts = df[col].dropna().value_counts()
        
        for original_val, _count in val_counts.items():
            norm_val = normalize_value(original_val)
            if not norm_val:
                continue
                
            # Since value_counts is sorted descending, the first time we see 
            # a normalized key, it's the most frequent variant.
            if norm_val not in col_index:
                col_index[norm_val] = str(original_val)
                
        index[col] = col_index
        
    return index


def fuzzy_match_category(
    user_value: Any, 
    column: str, 
    value_index: dict[str, dict[str, str]], 
    threshold: float = 85.0
) -> dict[str, Any] | None:
    """Match a user-provided string against known categories using fuzzy logic.
    
    Priority of matching:
    1. Exact match against original values (O(1))
    2. Normalized exact match (O(1))
    3. RapidFuzz similarity against canonical values (O(N))
    
    Args:
        user_value: The string provided by the user.
        column: The DataFrame column to match against.
        value_index: The pre-computed index from build_value_index().
        threshold: The minimum RapidFuzz ratio score (0-100) to accept a match. Defaults to 85.0.
        
    Returns:
        dict: {"original": user_value, "matched": canonical_value, "score": float, "method": str}
        None: If no match meets the threshold or the column is not indexed.
    """
    if pd.isna(user_value) or user_value is None:
        return None
        
    user_str = str(user_value)
    
    if column not in value_index or not value_index[column]:
        return None
        
    col_index = value_index[column]
    
    # 1. Exact match (case sensitive, unnormalized)
    # Check if user string exactly equals one of the canonical values
    for canonical_val in col_index.values():
        if user_str == canonical_val:
            return {"original": user_str, "matched": canonical_val, "score": 100.0, "method": "exact"}
            
    # 2. Normalized exact match
    norm_user = normalize_value(user_str)
    if not norm_user:
        return None
        
    if norm_user in col_index:
        return {"original": user_str, "matched": col_index[norm_user], "score": 100.0, "method": "normalized"}
        
    # 3. Fuzzy match
    best_match = None
    best_score = 0.0
    
    # Calculate a dynamic threshold based on the length of the normalized user string.
    # Approach: Hybrid percentage with length-aware scaling.
    # Why: A flat 85.0 threshold fails on short strings with a single typo (e.g. "2 bhk" vs 
    # "2 bkk" yields 80.0). We scale the threshold down to 80.0 for short strings (<= 5 chars) 
    # to permit exactly 1 character typo. For longer strings, the flat percentage threshold 
    # naturally permits 1-2 typos. We also strictly check that any numerical digits present 
    # in the string match exactly, preventing dangerous over-matches (e.g. "4 bhk" vs "2 bhk").
    user_len = len(norm_user)
    if user_len <= 5:
        effective_threshold = min(threshold, 80.0)
    else:
        effective_threshold = threshold
        
    user_digits = set(re.findall(r'\d+', norm_user))
    
    # We compare the normalized user string against the normalized index keys
    for norm_key, canonical_val in col_index.items():
        score = fuzz.ratio(norm_user, norm_key)
        
        # Prevent numbers from morphing during fuzzy match (e.g. "4" matching "2")
        canon_digits = set(re.findall(r'\d+', norm_key))
        if user_digits != canon_digits:
            continue
            
        if score >= effective_threshold:
            if score > best_score:
                best_score = score
                best_match = canonical_val
            elif score == best_score and best_match is not None:
                # Tie-breaker: choose the one closest in length to the original user string
                len_diff_current = abs(len(canonical_val) - len(user_str))
                len_diff_best = abs(len(best_match) - len(user_str))
                if len_diff_current < len_diff_best:
                    best_match = canonical_val

    if best_match is not None:
        return {"original": user_str, "matched": best_match, "score": round(best_score, 1), "method": "fuzzy"}
        
    return None
