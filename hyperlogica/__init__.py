"""
Hyperlogica: Hyperdimensional Vector-Based Reasoning System

A Python library that implements the AI Conceptual Exchange Protocol (ACEP)
for efficient AI-to-AI communication using hyperdimensional computing principles.
"""

__version__ = "0.1.0"

# Import core components for easy access
from .config_parser import parse_input_config, validate_config
from .vector_operations import (
    generate_vector, normalize_vector, bind_vectors, unbind_vectors, 
    bundle_vectors, calculate_similarity, cleanse_vector
)
from .vector_store import create_store, add_vector, get_vector, find_similar_vectors
from .llm_interface import convert_english_to_acep, convert_acep_to_english, generate_explanation
from .reasoning_engine import apply_modus_ponens, calculate_certainty, recalibrate_certainty
from .state_management import create_state, add_concept_to_state, add_relation_to_state
from .error_handling import success, error, is_success, is_error, get_value, get_error

# Main processing function
from .hyperlogica import process_input_file