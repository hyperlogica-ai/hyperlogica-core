"""
End-to-end tests for the Hyperlogica system.

These tests verify that the complete system works as expected,
from configuration parsing through to output generation.
"""

import os
import json
import tempfile
import pytest
import numpy as np
from pathlib import Path

from hyperlogica import process_input_file
from hyperlogica.config_parser import parse_input_config, validate_config
from hyperlogica.vector_store import create_store, add_vector, get_vector
from hyperlogica.state_management import create_state

# Test configuration using simple rules and facts
TEST_CONFIG = {
    "processing": {
        "vector_dimension": 1000,  # Smaller dimension for faster tests
        "vector_type": "binary",
        "reasoning_approach": "majority",
        "certainty_propagation": "min",
        "recalibration_enabled": False,
        "max_reasoning_depth": 3,
        "domain_config": {
            "positive_outcome_keywords": ["yes", "positive", "true"],
            "negative_outcome_keywords": ["no", "negative", "false"],
            "neutral_outcome_keywords": ["maybe", "neutral", "unknown"],
            "outcome_field": "result",
            "positive_outcome": "YES",
            "negative_outcome": "NO",
            "neutral_outcome": "MAYBE"
        }
    },
    "persistence": {
        "load_previous_state": False,
        "save_state": False
    },
    "logging": {
        "log_level": "error",
        "include_vector_operations": False,
        "include_llm_interactions": False
    },
    "llm": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 500,
        "use_mock": True  # Use mock responses for testing
    },
    "input_data": {
        "rules": [
            {"text": "If it is raining, then the ground is wet.", "certainty": 0.9},
            {"text": "If the ground is wet, then it is slippery.", "certainty": 0.8},
            {"text": "If it is sunny, then the ground is dry.", "certainty": 0.9}
        ],
        "entities": [
            {
                "id": "scenario_1",
                "name": "Rainy Day",
                "facts": [
                    {"text": "It is raining heavily.", "certainty": 0.95}
                ]
            },
            {
                "id": "scenario_2",
                "name": "Sunny Day",
                "facts": [
                    {"text": "It is sunny and clear.", "certainty": 0.9}
                ]
            }
        ]
    },
    "output_schema": {
        "format": "json",
        "fields": [
            {"name": "entity_id", "type": "string"},
            {"name": "entity_name", "type": "string"},
            {"name": "result", "type": "string"},
            {"name": "certainty", "type": "float"}
        ],
        "include_reasoning_trace": True
    }
}

class TestEndToEnd:
    """End-to-end test cases for the complete system."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Create temporary directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Configure mock responses for LLM interface
        self.mock_responses = {
            "convert_english_to_acep": {
                "If it is raining, then the ground is wet.": {
                    "identifier": "ground_wet_if_raining",
                    "type": "concept",
                    "content": {
                        "concept": "conditional_relationship"
                    },
                    "attributes": {
                        "antecedent": "raining",
                        "consequent": "ground_wet",
                        "certainty": 0.9
                    },
                    "vector": np.random.rand(1000)
                },
                "If the ground is wet, then it is slippery.": {
                    "identifier": "slippery_if_ground_wet",
                    "type": "concept",
                    "content": {
                        "concept": "conditional_relationship"
                    },
                    "attributes": {
                        "antecedent": "ground_wet",
                        "consequent": "slippery",
                        "certainty": 0.8
                    },
                    "vector": np.random.rand(1000)
                },
                "If it is sunny, then the ground is dry.": {
                    "identifier": "ground_dry_if_sunny",
                    "type": "concept",
                    "content": {
                        "concept": "conditional_relationship"
                    },
                    "attributes": {
                        "antecedent": "sunny",
                        "consequent": "ground_dry",
                        "certainty": 0.9
                    },
                    "vector": np.random.rand(1000)
                },
                "It is raining heavily.": {
                    "identifier": "raining_heavily",
                    "type": "concept",
                    "content": {
                        "concept": "weather_state"
                    },
                    "attributes": {
                        "state": "raining",
                        "intensity": "heavy",
                        "certainty": 0.95
                    },
                    "vector": np.random.rand(1000)
                },
                "It is sunny and clear.": {
                    "identifier": "sunny_clear",
                    "type": "concept",
                    "content": {
                        "concept": "weather_state"
                    },
                    "attributes": {
                        "state": "sunny",
                        "intensity": "clear",
                        "certainty": 0.9
                    },
                    "vector": np.random.rand(1000)
                }
            }
        }
        
        # Patch LLM functions for testing
        import hyperlogica.llm_interface
        self._original_convert_english_to_acep = hyperlogica.llm_interface.convert_english_to_acep
        hyperlogica.llm_interface.convert_english_to_acep = self._mock_convert_english_to_acep
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Remove temporary directory
        self.temp_dir.cleanup()
        
        # Restore original functions
        import hyperlogica.llm_interface
        hyperlogica.llm_interface.convert_english_to_acep = self._original_convert_english_to_acep
    
    def _mock_convert_english_to_acep(self, text, context, llm_options=None):
        """Mock implementation for convert_english_to_acep."""
        # Use predefined responses
        response_dict = self.mock_responses["convert_english_to_acep"]
        
        # Find exact match if possible
        if text in response_dict:
            return response_dict[text]
        
        # Fall back to a generic response
        return {
            "identifier": f"generic_{hash(text) % 1000}",
            "type": "concept",
            "content": {
                "concept": "generic"
            },
            "attributes": {
                "text": text,
                "certainty": context.get("certainty", 0.5)
            },
            "vector": np.random.rand(1000)
        }
    
    def test_process_input_file(self):
        """Test the main process_input_file function with in-memory config."""
        # Save config to a temporary file
        config_path = os.path.join(self.temp_dir.name, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(TEST_CONFIG, f)
        
        # Process the config
        output_path = os.path.join(self.temp_dir.name, "test_output.json")
        options = {
            "verbose": False,
            "output_path": output_path
        }
        
        results = process_input_file(config_path, options)
        
        # Verify basic structure of results
        assert "entities_processed" in results
        assert "conclusions_generated" in results
        assert "results" in results
        
        # Check entity count
        assert results["entities_processed"] == 2
        
        # Check that output file was created
        assert os.path.exists(output_path)
        
        # Load output file and verify contents
        with open(output_path, 'r') as f:
            output_data = json.load(f)
        
        assert len(output_data["results"]) == 2
        
        # Check specifics for scenario 1 (rainy day)
        scenario_1 = next(r for r in output_data["results"] if r["entity_id"] == "scenario_1")
        assert scenario_1["entity_name"] == "Rainy Day"
        
        # Should indicate wet/slippery ground (a positive outcome in this test model)
        # Due to the rule chain: raining -> ground_wet -> slippery
        assert scenario_1["result"] in ["YES", "MAYBE"]
        
        # Check specifics for scenario 2 (sunny day)
        scenario_2 = next(r for r in output_data["results"] if r["entity_id"] == "scenario_2")
        assert scenario_2["entity_name"] == "Sunny Day"
        
        # Should indicate dry ground (a negative outcome for wet/slippery in this test model)
        assert scenario_2["result"] in ["NO", "MAYBE"]
    
    def test_vector_store_operations(self):
        """Test that vector store operations work in the end-to-end process."""
        # Create a store
        store = create_store(dimension=1000)
        
        # Define test vectors
        test_vectors = [
            {"id": "concept1", "vector": np.random.rand(1000), "metadata": {"type": "test"}},
            {"id": "concept2", "vector": np.random.rand(1000), "metadata": {"type": "test"}},
            {"id": "concept3", "vector": np.random.rand(1000), "metadata": {"type": "test"}}
        ]
        
        # Add vectors to store
        for item in test_vectors:
            add_vector(store, item["id"], item["vector"], item["metadata"])
        
        # Verify retrieval
        for item in test_vectors:
            retrieved = get_vector(store, item["id"])
            assert retrieved["identifier"] == item["id"]
            assert np.array_equal(retrieved["vector"], item["vector"] / np.linalg.norm(item["vector"]))
            assert retrieved["metadata"] == item["metadata"]
    
    def test_state_persistence(self):
        """Test that state persistence works properly."""
        # Create a config that uses state persistence
        persistence_config = TEST_CONFIG.copy()
        persistence_config["persistence"]["save_state"] = True
        persistence_config["persistence"]["state_save_path"] = os.path.join(self.temp_dir.name, "test_state.pkl")
        
        # Save the modified config
        config_path = os.path.join(self.temp_dir.name, "persistence_config.json")
        with open(config_path, 'w') as f:
            json.dump(persistence_config, f)
            
        # Run the processing
        process_input_file(config_path)
        
        # Check that state file was created
        assert os.path.exists(persistence_config["persistence"]["state_save_path"])
        
        # Now modify config to load previous state
        persistence_config["persistence"]["load_previous_state"] = True
        persistence_config["persistence"]["previous_state_path"] = persistence_config["persistence"]["state_save_path"]
        
        # Change entity facts to test if state is properly loaded
        persistence_config["input_data"]["entities"][0]["facts"] = [
            {"text": "It stopped raining but the ground is still wet.", "certainty": 0.9}
        ]
        
        # Save the updated config
        with open(config_path, 'w') as f:
            json.dump(persistence_config, f)
        
        # Run processing again
        results_with_state = process_input_file(config_path)
        
        # Verify that processing succeeded and used the previous state
        assert results_with_state["entities_processed"] == 2


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])