"""
Test vector operations module
"""

import numpy as np
import pytest
from hyperlogica.vector_operations import (
    generate_vector, normalize_vector, bind_vectors, unbind_vectors,
    bundle_vectors, calculate_similarity, create_role_vectors,
    create_conditional_representation, create_fact_representation
)

def test_generate_vector():
    """Test vector generation is deterministic and correct dimension"""
    # Generate a vector
    dim = 1000
    vector = generate_vector("test", dim)
    
    # Check dimension
    assert vector.shape == (dim,)
    
    # Check determinism
    vector2 = generate_vector("test", dim)
    assert np.array_equal(vector, vector2)
    
    # Different text should produce different vector
    vector3 = generate_vector("test2", dim)
    assert not np.array_equal(vector, vector3)

def test_normalize_vector():
    """Test vector normalization"""
    # Create a random vector
    vector = np.array([1.0, 2.0, 3.0])
    
    # Normalize
    norm_vector = normalize_vector(vector)
    
    # Check length is 1
    assert np.isclose(np.linalg.norm(norm_vector), 1.0)
    
    # Test with zeros vector
    zero_vector = np.zeros(3)
    with pytest.raises(ValueError):
        normalize_vector(zero_vector)

def test_bind_vectors():
    """Test binding operation"""
    # Create binary vectors
    v1 = np.array([0, 1, 0, 1])
    v2 = np.array([1, 0, 1, 0])
    
    # Bind using XOR
    bound = bind_vectors(v1, v2, method="xor")
    
    # Expected result: [1, 1, 1, 1]
    expected = np.array([1, 1, 1, 1])
    assert np.array_equal(bound, expected)
    
    # Test auto-detection
    bound_auto = bind_vectors(v1, v2)
    assert np.array_equal(bound_auto, expected)
    
    # Test with continuous vectors
    v3 = np.array([0.5, 0.5, 0.5])
    v4 = np.array([0.1, 0.2, 0.3])
    bound_cont = bind_vectors(v3, v4, method="convolution")
    assert bound_cont.shape == v3.shape

def test_unbind_vectors():
    """Test unbinding operation"""
    # Create binary vectors
    v1 = np.array([0, 1, 0, 1])
    v2 = np.array([1, 0, 1, 0])
    
    # Bind using XOR
    bound = bind_vectors(v1, v2, method="xor")
    
    # Unbind to get v2 from bound and v1
    recovered = unbind_vectors(bound, v1, method="xor")
    
    # Should recover v2
    assert np.array_equal(recovered, v2)
    
    # Test auto-detection
    recovered_auto = unbind_vectors(bound, v1)
    assert np.array_equal(recovered_auto, v2)

def test_bundle_vectors():
    """Test bundling operation"""
    # Create binary vectors
    v1 = np.array([0, 1, 0, 1])
    v2 = np.array([1, 0, 1, 0])
    
    # Bundle with equal weights
    bundled = bundle_vectors([v1, v2])
    
    # With binary vectors, result should have 1s where either has 1
    expected = np.array([1, 1, 1, 1])
    assert np.array_equal(bundled, expected)
    
    # Test with weights
    bundled_weighted = bundle_vectors([v1, v2], weights=[0.8, 0.2])
    # v1 should dominate, so expect closer to v1
    assert bundled_weighted[1] == 1  # From v1
    assert bundled_weighted[2] == 0  # From v1

def test_calculate_similarity():
    """Test similarity calculation"""
    # Create vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([0.9, 0.1, 0.0])
    
    # Orthogonal vectors should have 0.5 similarity (cosine mapped to [0,1])
    assert np.isclose(calculate_similarity(v1, v2, method="cosine"), 0.5)
    
    # Similar vectors should have high similarity
    assert calculate_similarity(v1, v3, method="cosine") > 0.9
    
    # Same vector should have 1.0 similarity
    assert np.isclose(calculate_similarity(v1, v1, method="cosine"), 1.0)

def test_create_role_vectors():
    """Test creation of role vectors"""
    # Create role vectors
    dim = 1000
    roles = create_role_vectors(dim)
    
    # Check key roles exist
    assert "condition" in roles
    assert "implication" in roles
    assert "concept" in roles
    
    # All vectors should be normalized
    for role, vector in roles.items():
        assert np.isclose(np.linalg.norm(vector), 1.0)
    
    # Vectors should be approximately orthogonal
    similarity = calculate_similarity(roles["condition"], roles["implication"])
    assert 0.4 < similarity < 0.6  # Approximately orthogonal (0.5 for orthogonal after mapping to [0,1])

def test_create_conditional_representation():
    """Test creation of conditional (rule) representation"""
    # Create role vectors
    dim = 1000
    roles = create_role_vectors(dim)
    
    # Create condition and implication
    condition = {
        "concept": "pe_ratio",
        "relation": "below",
        "reference": "industry_average"
    }
    
    implication = {
        "concept": "valuation",
        "state": "undervalued"
    }
    
    # Create representation
    representation = create_conditional_representation(condition, implication, roles, dim)
    
    # Check required vectors exist
    assert "condition_vector" in representation
    assert "implication_vector" in representation
    assert "rule_vector" in representation
    assert "component_vectors" in representation
    
    # All vectors should be normalized
    assert np.isclose(np.linalg.norm(representation["condition_vector"]), 1.0)
    assert np.isclose(np.linalg.norm(representation["implication_vector"]), 1.0)
    assert np.isclose(np.linalg.norm(representation["rule_vector"]), 1.0)

def test_create_fact_representation():
    """Test creation of fact representation"""
    # Create role vectors
    dim = 1000
    roles = create_role_vectors(dim)
    
    # Create fact content
    fact = {
        "concept": "pe_ratio",
        "relation": "below",
        "reference": "industry_average",
        "actual_value": 15.2,
        "reference_value": 20.5
    }
    
    # Create representation
    representation = create_fact_representation(fact, roles, dim)
    
    # Check required vectors exist
    assert "fact_vector" in representation
    assert "component_vectors" in representation
    
    # All vectors should be normalized
    assert np.isclose(np.linalg.norm(representation["fact_vector"]), 1.0)
