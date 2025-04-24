"""
Tests for the LLM interface module.

These tests verify that the LLM interface correctly handles
various text conversion operations and API interactions.
"""

import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hyperlogica.llm_interface import (
    create_english_to_acep_prompt,
    create_acep_to_english_prompt,
    call_openai_api,
    convert_english_to_acep,
    convert_acep_to_english,
    generate_explanation,
    create_embedding,
    generate_deterministic_vector,
    create_normalized_identifier
)

class TestLLMInterface:
    """Tests for the LLM interface module."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Ensure tests don't make actual API calls
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            os.environ.pop("OPENAI_API_KEY")
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Restore API key if it was set
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
    
    def test_create_english_to_acep_prompt(self):
        """Test creating prompts for English to ACEP conversion."""
        # Test with basic text
        text = "If it rains, the ground gets wet."
        context = {
            "domain": "weather",
            "entity_id": "scenario1"
        }
        
        prompt = create_english_to_acep_prompt(text, context)
        
        # Check that the prompt contains the text and context
        assert text in prompt
        assert "weather" in prompt
        assert "scenario1" in prompt
        
        # Test with empty context
        prompt_empty_context = create_english_to_acep_prompt(text, {})
        assert text in prompt_empty_context
    
    def test_create_acep_to_english_prompt(self):
        """Test creating prompts for ACEP to English conversion."""
        # Create a sample ACEP representation
        acep_representation = {
            "identifier": "ground_wet_if_rain",
            "type": "concept",
            "content": {
                "concept": "conditional_relationship"
            },
            "attributes": {
                "antecedent": "rain",
                "consequent": "ground_wet",
                "certainty": 0.9
            }
        }
        
        context = {
            "domain": "weather",
            "entity_id": "scenario1"
        }
        
        prompt = create_acep_to_english_prompt(acep_representation, context)
        
        # Check that the prompt contains the representation and context
        assert json.dumps(acep_representation) in prompt
        assert "weather" in prompt
        assert "scenario1" in prompt
    
    @patch("hyperlogica.llm_interface.openai.ChatCompletion.create")
    def test_call_openai_api(self, mock_create):
        """Test calling the OpenAI API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = {"content": "Test response"}
        mock_create.return_value = mock_response
        
        # Call API
        prompt = "Test prompt"
        model = "gpt-3.5-turbo"
        options = {"temperature": 0.0, "max_tokens": 500}
        
        response = call_openai_api(prompt, model, options)
        
        # Verify API was called with correct parameters
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        assert kwargs["model"] == model
        assert kwargs["messages"][0]["content"] == prompt
        assert kwargs["temperature"] == options["temperature"]
        assert kwargs["max_tokens"] == options["max_tokens"]
        
        # Verify response
        assert response.choices[0].message["content"] == "Test response"
    
    @patch("hyperlogica.llm_interface.call_openai_api")
    def test_convert_english_to_acep(self, mock_call_api):
        """Test converting English text to ACEP representation."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = {"content": json.dumps({
            "identifier": "ground_wet_if_rain",
            "type": "concept",
            "content": {
                "concept": "conditional_relationship"
            },
            "attributes": {
                "antecedent": "rain",
                "consequent": "ground_wet",
                "certainty": 0.9
            }
        })}
        mock_call_api.return_value = mock_response
        
        # Convert text
        text = "If it rains, the ground gets wet."
        context = {"domain": "weather"}
        llm_options = {"model": "gpt-3.5-turbo"}
        
        result = convert_english_to_acep(text, context, llm_options)
        
        # Verify API was called
        mock_call_api.assert_called_once()
        
        # Verify result structure
        assert result["identifier"] == "ground_wet_if_rain"
        assert result["type"] == "concept"
        assert result["attributes"]["antecedent"] == "rain"
        assert result["attributes"]["certainty"] == 0.9
    
    @patch("hyperlogica.llm_interface.call_openai_api")
    def test_convert_acep_to_english(self, mock_call_api):
        """Test converting ACEP representation to English text."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = {"content": "If it rains, then the ground will get wet."}
        mock_call_api.return_value = mock_response
        
        # Create ACEP representation
        acep_representation = {
            "identifier": "ground_wet_if_rain",
            "type": "concept",
            "content": {
                "concept": "conditional_relationship"
            },
            "attributes": {
                "antecedent": "rain",
                "consequent": "ground_wet",
                "certainty": 0.9
            }
        }
        
        context = {"domain": "weather"}
        llm_options = {"model": "gpt-3.5-turbo"}
        
        result = convert_acep_to_english(acep_representation, context, llm_options)
        
        # Verify API was called
        mock_call_api.assert_called_once()
        
        # Verify result
        assert result == "If it rains, then the ground will get wet."
    
    @patch("hyperlogica.llm_interface.call_openai_api")
    def test_generate_explanation(self, mock_call_api):
        """Test generating explanations from reasoning traces."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = {"content": "The ground is wet because it is raining."}
        mock_call_api.return_value = mock_response
        
        # Create reasoning trace
        reasoning_trace = {
            "session_id": "test_session",
            "steps": [
                {
                    "step_id": 1,
                    "pattern": "modus_ponens",
                    "premises": ["raining", "rain_causes_wet_ground"],
                    "conclusion": "ground_wet",
                    "certainty": 0.9
                }
            ],
            "final_conclusions": [
                {
                    "identifier": "ground_wet",
                    "text": "The ground is wet",
                    "certainty": 0.9
                }
            ]
        }
        
        context = {
            "domain": "weather",
            "recommendation": "YES",
            "certainty": 0.9
        }
        
        llm_options = {"model": "gpt-3.5-turbo"}
        
        result = generate_explanation(reasoning_trace, context, llm_options)
        
        # Verify API was called
        mock_call_api.assert_called_once()
        
        # Verify result
        assert result == "The ground is wet because it is raining."
    
    @patch("hyperlogica.llm_interface.openai.Embedding.create")
    def test_create_embedding(self, mock_create):
        """Test creating embeddings using OpenAI API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_create.return_value = mock_response
        
        # Create embedding
        text = "This is a test"
        embedding, metadata = create_embedding(text)
        
        # Verify API was called
        mock_create.assert_called_once()
        
        # Verify result
        assert len(embedding) == 3
        assert np.array_equal(embedding, np.array([0.1, 0.2, 0.3]))
        assert "dimensions" in metadata
        assert metadata["dimensions"] == 3
    
    def test_generate_deterministic_vector(self):
        """Test generating deterministic vectors from text."""
        # Generate vectors for the same text
        text = "This is a test"
        dimension = 1000
        
        vector1 = generate_deterministic_vector(text, dimension)
        vector2 = generate_deterministic_vector(text, dimension)
        
        # Vectors for the same text should be identical
        assert np.array_equal(vector1, vector2)
        
        # Generate vector for different text
        different_text = "This is different"
        different_vector = generate_deterministic_vector(different_text, dimension)
        
        # Vectors for different text should be different
        assert not np.array_equal(vector1, different_vector)
        
        # Check dimensions
        assert vector1.shape[0] == dimension
        
        # Check normalization
        assert np.isclose(np.linalg.norm(vector1), 1.0)
    
    def test_create_normalized_identifier(self):
        """Test creating normalized identifiers from text."""
        # Test basic normalization
        text = "This is a Test!"
        identifier = create_normalized_identifier(text)
        assert identifier == "this_is_a_test"
        
        # Test with special characters
        text = "Complex: example! with $special& characters."
        identifier = create_normalized_identifier(text)
        assert identifier == "complex_example_with_special_characters"
        
        # Test truncation of long text
        long_text = "This is a very long text that should be truncated" * 5
        identifier = create_normalized_identifier(long_text)
        assert len(identifier) <= 50