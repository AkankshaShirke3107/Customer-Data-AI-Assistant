import pytest
import json
from unittest.mock import patch, MagicMock
from models import RootIntentModel
from gemini_helper import understand_intent
from pydantic import ValidationError

@patch("gemini_helper._get_client")
def test_understand_intent_success(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Return a valid JSON matching RootIntentModel
    mock_response.text = '{"operation": "count", "conditions": []}'
    mock_client.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client
    
    intent = understand_intent("how many properties?", [], {})
    assert isinstance(intent, RootIntentModel)
    assert intent.operation == "count"
    assert len(intent.conditions) == 0

@patch("gemini_helper._get_client")
def test_understand_intent_retry_on_validation_error(mock_get_client):
    mock_client = MagicMock()
    # First response: invalid operation (ValidationError)
    mock_response_bad = MagicMock()
    mock_response_bad.text = '{"operation": "invalid_op_xyz", "conditions": []}'
    
    # Second response: valid
    mock_response_good = MagicMock()
    mock_response_good.text = '{"operation": "count", "conditions": []}'
    
    mock_client.generate_content.side_effect = [mock_response_bad, mock_response_good]
    mock_get_client.return_value = mock_client
    
    intent = understand_intent("count properties", [], {})
    assert isinstance(intent, RootIntentModel)
    assert intent.operation == "count"
    assert mock_client.generate_content.call_count == 2
    
    # Check that the second call contained the error message
    args, kwargs = mock_client.generate_content.call_args
    prompt = args[0]
    assert "Failed JSON Output:" in prompt
    assert "invalid_op_xyz" in prompt
    assert "Validation Error:" in prompt

@patch("gemini_helper._get_client")
def test_understand_intent_fails_after_retries(mock_get_client):
    mock_client = MagicMock()
    mock_response_bad = MagicMock()
    mock_response_bad.text = '{"operation": "invalid_op_xyz", "conditions": []}'
    
    # Always return bad JSON
    mock_client.generate_content.return_value = mock_response_bad
    mock_get_client.return_value = mock_client
    
    with pytest.raises(ValueError, match="Failed to generate valid intent after"):
        understand_intent("count properties", [], {})
