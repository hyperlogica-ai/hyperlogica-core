"""
Test reasoning engine module
"""

import numpy as np
import pytest
from hyperlogica.vector_operations import (
    create_role_vectors, create_conditional_representation, create_fact_representation
)
from hyperlogica.reasoning_engine import (
    match_condition_to_fact, create_conclusion, apply_vector_chain_reasoning,
    classify_conclusion, generate_explanation
)

def test_match_condition_to_fact():
    """Test matching condition to fact"""
    # Create role vectors
    dim = 1000
    roles = create_role_vectors(dim)
    
    # Create a rule
    condition = {
        "concept": "pe_ratio",
        "relation": "below",
        "reference": "industry_average"
    }
    
    implication = {
        "concept": "valuation",
        "state": "undervalued"
    }
    
    rule_vectors = create_conditional_representation(condition, implication, roles, dim)
    
    rule = {
        "identifier": "test_rule",
        "condition_vector": rule_vectors["condition_vector"],
        "implication_vector": rule_vectors["implication_vector"],
        "vector": rule_vectors["rule_vector"],
        "attributes": {"certainty": 0.9}
    }
    
    # Create a matching fact
    fact_content = {
        "concept": "pe_ratio",
        "relation": "below",
        "reference": "industry_average"
    }
    
    fact_vectors = create_fact_representation(fact_content, roles, dim)
    
    fact = {
        "identifier": "test_fact",
        "vector": fact_vectors["fact_vector"],
        "attributes": {"certainty": 0.8}
    }
    
    # Test matching
    match_result, similarity = match_condition_to_fact(rule, fact, roles)
    
    # Should be a match with high similarity
    assert match_result is True
    assert similarity > 0.7
    
    # Create a non-matching fact
    non_matching_content = {
        "concept": "pe_ratio",
        "relation": "above",  # Different relation
        "reference": "industry_average"
    }
    
    non_matching_vectors = create_fact_representation(non_matching_content, roles, dim)
    
    non_matching_fact = {
        "identifier": "non_matching_fact",
        "vector": non_matching_vectors["fact_vector"],
        "attributes": {"certainty": 0.8}
    }
    
    # Test non-matching
    match_result, similarity = match_condition_to_fact(rule, non_matching_fact, roles)
    
    # Should not be a match or have low similarity
    assert match_result is False or similarity < 0.7

def test_create_conclusion():
    """Test creating a conclusion from a rule and a fact"""
    # Create role vectors
    dim = 1000
    roles = create_role_vectors(dim)
    
    # Create a rule
    condition = {
        "concept": "pe_ratio",
        "relation": "below",
        "reference": "industry_average"
    }
    
    implication = {
        "concept": "valuation",
        "state": "undervalued"
    }
    
    rule_vectors = create_conditional_representation(condition, implication, roles, dim)
    
    rule = {
        "identifier": "test_rule",
        "condition_vector": rule_vectors["condition_vector"],
        "implication_vector": rule_vectors["implication_vector"],
        "vector": rule_vectors["rule_vector"],
        "attributes": {"certainty": 0.9},
        "acep": {
            "content": {
                "implication": implication
            }
        }
    }
    
    # Create a matching fact
    fact_content = {
        "concept": "pe_ratio",
        "relation": "below",
        "reference": "industry_average"
    }
    
    fact_vectors = create_fact_representation(fact_content, roles, dim)
    
    fact = {
        "identifier": "test_fact",
        "vector": fact_vectors["fact_vector"],
        "attributes": {"certainty": 0.8}
    }
    
    # Create conclusion
    similarity = 0.85
    entity_id = "TEST_ENTITY"
    conclusion = create_conclusion(rule, fact, similarity, entity_id, roles)
    
    # Check conclusion properties
    assert "identifier" in conclusion
    assert "vector" in conclusion
    assert "attributes" in conclusion
    assert "acep" in conclusion
    
    # Check entity_id is set
    assert conclusion["attributes"]["entity_id"] == entity_id
    
    # Check certainty calculation (min of rule, fact, and match)
    expected_certainty = min(0.9, 0.8) * similarity
    assert np.isclose(conclusion["attributes"]["certainty"], expected_certainty)
    
    # Check derived_from includes rule and fact
    assert rule["identifier"] in conclusion["attributes"]["derived_from"]
    assert fact["identifier"] in conclusion["attributes"]["derived_from"]
    
    # Check ACEP structure
    assert conclusion["acep"]["type"] == "derived_assertion"
    assert "concept" in conclusion["acep"]["content"]
    assert "state" in conclusion["acep"]["content"]

def test_classify_conclusion():
    """Test classification of conclusion as positive/negative/neutral"""
    # Create a positive conclusion (undervalued is positive)
    positive_conclusion = {
        "acep": {
            "content": {
                "concept": "valuation",
                "state": "undervalued"
            }
        },
        "attributes": {}
    }
    
    # Test classification
    assert classify_conclusion(positive_conclusion) == "positive"
    
    # Create a negative conclusion (overvalued is negative)
    negative_conclusion = {
        "acep": {
            "content": {
                "concept": "valuation",
                "state": "overvalued"
            }
        },
        "attributes": {}
    }
    
    # Test classification
    assert classify_conclusion(negative_conclusion) == "negative"
    
    # Create a neutral conclusion
    neutral_conclusion = {
        "acep": {
            "content": {
                "concept": "valuation",
                "state": "neutral"
            }
        },
        "attributes": {}
    }
    
    # Test classification
    assert classify_conclusion(neutral_conclusion) == "neutral"
    
    # Test with explicit signal type in attributes
    explicit_conclusion = {
        "acep": {
            "content": {
                "concept": "valuation",
                "state": "neutral"  # Would be neutral
            }
        },
        "attributes": {
            "signal_type": "positive"  # But explicitly marked positive
        }
    }
    
    # Explicit should override content-based classification
    assert classify_conclusion(explicit_conclusion) == "positive"

def test_generate_explanation():
    """Test explanation generation"""
    # Create a simple reasoning result
    reasoning_result = {
        "outcome": "POSITIVE", 
        "certainty": 0.85,
        "entity_id": "TEST_ENTITY",
        "positive_conclusions": [
            {
                "acep": {
                    "content": {
                        "concept": "valuation",
                        "state": "undervalued"
                    }
                },
                "attributes": {
                    "certainty": 0.9
                }
            },
            {
                "acep": {
                    "content": {
                        "concept": "company",
                        "state": "growth_phase"
                    }
                },
                "attributes": {
                    "certainty": 0.8
                }
            }
        ],
        "negative_conclusions": [
            {
                "acep": {
                    "content": {
                        "concept": "stock_performance",
                        "state": "potentially_overvalued"
                    }
                },
                "attributes": {
                    "certainty": 0.6
                }
            }
        ],
        "chains": [
            {
                "steps": [
                    {
                        "step_number": 1,
                        "certainty": 0.9,
                        "acep": {
                            "content": {
                                "concept": "valuation",
                                "state": "undervalued"
                            }
                        }
                    }
                ]
            }
        ],
        "evidence_weights": {
            "positive": 1.7,
            "negative": 0.6,
            "neutral": 0.0
        }
    }
    
    # Generate explanation
    explanation = generate_explanation(reasoning_result)
    
    # Basic checks
    assert explanation  # Not empty
    assert "TEST_ENTITY" in explanation  # Includes entity ID
    assert "85%" in explanation  # Includes certainty percentage
    assert "BUY" in explanation or "POSITIVE" in explanation  # Mentions recommendation
    assert "valuation" in explanation  # Mentions key concept
    assert "undervalued" in explanation  # Mentions key state
