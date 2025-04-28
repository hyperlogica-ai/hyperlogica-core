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
    bundle_vectors, calculate_similarity
)
from .vector_store import create_store, add_vector, get_vector, find_similar_vectors
from .error_handling import success, error, is_success, is_error, get_value, get_error

# Import LLM interface components
from .llm_interface import (
    convert_english_to_acep, convert_acep_to_english, 
    generate_explanation, create_embedding
)

# Main processing function
from .hyperlogica import process_input_file

# Ontology mapping components
from .ontology_mapper import (
    create_ontology_mapper, map_text_to_ontology,
    standardize_facts_with_ontology, enhance_reasoning_with_ontology
)
