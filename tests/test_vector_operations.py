"""
Test suite for the vector_operations module using a functional testing approach.
Each test function is independent and follows the same functional style as the
production code, avoiding classes and shared state.
"""

import numpy as np
from vector_operations import (
    generate_vector, normalize_vector, bind_vectors, unbind_vectors,
    bundle_vectors, permute_vector, calculate_similarity, create_hd_index,
    find_similar_vectors
)
import pytest

# Test vector generation with different parameters
def test_generate_vector_deterministic():
    """Test that vector generation is deterministic with the same seed."""
    v1 = generate_vector("test", dimension=100, seed=42)
    v2 = generate_vector("test", dimension=100, seed=42)
    assert np.array_equal(v1, v2), "Vectors with same seed should be identical"

def test_generate_vector_different_seeds():
    """Test that different seeds produce different vectors."""
    v1 = generate_vector("test", dimension=100, seed=42)
    v3 = generate_vector("test", dimension=100, seed=43)
    assert not np.array_equal(v1, v3), "Vectors with different seeds should differ"

def test_generate_vector_different_text():
    """Test that different text with same seed produces different vectors."""
    v1 = generate_vector("test", dimension=100, seed=42)
    v4 = generate_vector("different", dimension=100, seed=42)
    assert not np.array_equal(v1, v4), "Vectors from different text with same seed should differ"

def test_generate_vector_dimensions():
    """Test vector dimensions."""
    v5 = generate_vector("test", dimension=200)
    assert len(v5) == 200, "Vector should have the specified dimension"

def test_generate_vector_binary():
    """Test binary vectors contain only 0s and 1s."""
    v6 = generate_vector("test", dimension=100, vector_type="binary")
    assert np.all(np.logical_or(v6 == 0, v6 == 1)), "Binary vector should contain only 0s and 1s"

def test_generate_vector_bipolar():
    """Test bipolar vectors contain only -1s and 1s."""
    v7 = generate_vector("test", dimension=100, vector_type="bipolar")
    assert np.all(np.logical_or(v7 == -1, v7 == 1)), "Bipolar vector should contain only -1s and 1s"

def test_generate_vector_continuous():
    """Test continuous vectors are normalized."""
    v8 = generate_vector("test", dimension=100, vector_type="continuous")
    assert np.isclose(np.linalg.norm(v8), 1.0), "Continuous vector should be normalized to unit length"

def test_generate_vector_invalid_dimension():
    """Test invalid dimension raises ValueError."""
    with pytest.raises(ValueError):
        generate_vector("test", dimension=0)

def test_generate_vector_invalid_type():
    """Test invalid vector type raises ValueError."""
    with pytest.raises(ValueError):
        generate_vector("test", vector_type="invalid")

# Test vector normalization
def test_normalize_vector_unit_length():
    """Test normalization results in unit length."""
    v = np.random.rand(100)
    v_norm = normalize_vector(v)
    assert np.isclose(np.linalg.norm(v_norm), 1.0), "Normalized vector should have unit length"

def test_normalize_vector_preserves_direction():
    """Test normalization preserves direction."""
    v = np.random.rand(100)
    v_norm = normalize_vector(v)
    dot_product = np.dot(v, v_norm) / np.linalg.norm(v)
    assert np.isclose(dot_product, 1.0), "Normalization should preserve vector direction"

def test_normalize_already_normalized():
    """Test normalizing an already normalized vector."""
    v = np.random.rand(100)
    v_norm = normalize_vector(v)
    v_norm_again = normalize_vector(v_norm)
    assert np.isclose(np.linalg.norm(v_norm_again), 1.0), "Normalizing an already normalized vector should not change it"

def test_normalize_zero_vector():
    """Test normalization of zero vector raises ValueError."""
    with pytest.raises(ValueError):
        normalize_vector(np.zeros(100))

# Test vector binding operations
def test_bind_vectors_binary_xor():
    """Test binding binary vectors with XOR."""
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 0, 1])
    result = bind_vectors(v1, v2, method="xor")
    expected = np.array([1, 0, 0, 1, 0])
    assert np.array_equal(result, expected), "XOR binding should work correctly for binary vectors"

def test_bind_vectors_bipolar_multiply():
    """Test binding bipolar vectors with multiplication."""
    v1 = np.array([1, -1, 1, -1, 1])
    v2 = np.array([-1, -1, 1, 1, 1])
    result = bind_vectors(v1, v2, method="multiply")
    expected = np.array([-1, 1, 1, -1, 1])
    assert np.array_equal(result, expected), "Multiply binding should work correctly for bipolar vectors"

def test_bind_vectors_auto_detect():
    """Test auto-detection of binding method."""
    # Binary vectors
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 0, 1])
    result_auto = bind_vectors(v1, v2, method="auto")
    result_xor = bind_vectors(v1, v2, method="xor")
    assert np.array_equal(result_auto, result_xor), "Auto method should detect binary vectors and use XOR"
    
    # Bipolar vectors
    v3 = np.array([1, -1, 1, -1, 1])
    v4 = np.array([-1, -1, 1, 1, 1])
    result_auto = bind_vectors(v3, v4, method="auto")
    result_multiply = bind_vectors(v3, v4, method="multiply")
    assert np.array_equal(result_auto, result_multiply), "Auto method should detect bipolar vectors and use multiply"

def test_bind_vectors_convolution():
    """Test binding continuous vectors with convolution."""
    # Create continuous unit vectors
    v1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    v2 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    
    # Bind with convolution
    result = bind_vectors(v1, v2, method="convolution")
    
    # Convolutional binding should produce a normalized vector different from inputs
    assert np.isclose(np.linalg.norm(result), 1.0), "Convolutional binding should produce a unit vector"
    assert not np.allclose(result, v1), "Bound vector should differ from first input"
    assert not np.allclose(result, v2), "Bound vector should differ from second input"

def test_bind_vectors_different_shapes():
    """Test binding vectors with different shapes raises ValueError."""
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        bind_vectors(v1, v2)

def test_bind_vectors_invalid_method():
    """Test binding with invalid method raises ValueError."""
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 0, 1])
    with pytest.raises(ValueError):
        bind_vectors(v1, v2, method="invalid")

# Test vector unbinding operations
def test_unbind_vectors_binary_xor():
    """Test unbinding binary vectors with XOR."""
    v_a = np.array([0, 1, 0, 1, 1])
    v_b = np.array([1, 1, 0, 0, 1])
    bound = bind_vectors(v_a, v_b, method="xor")
    
    # Recover v_b using v_a
    recovered_b = unbind_vectors(bound, v_a, method="xor")
    assert np.array_equal(recovered_b, v_b), "Should recover original vector v_b"
    
    # Recover v_a using v_b
    recovered_a = unbind_vectors(bound, v_b, method="xor")
    assert np.array_equal(recovered_a, v_a), "Should recover original vector v_a"

def test_unbind_vectors_bipolar_multiply():
    """Test unbinding bipolar vectors with multiplication."""
    v_a = np.array([1, -1, 1, -1, 1])
    v_b = np.array([-1, -1, 1, 1, 1])
    bound = bind_vectors(v_a, v_b, method="multiply")
    
    # Recover v_b using v_a
    recovered_b = unbind_vectors(bound, v_a, method="multiply")
    assert np.array_equal(recovered_b, v_b), "Should recover original vector v_b"
    
    # Recover v_a using v_b
    recovered_a = unbind_vectors(bound, v_b, method="multiply")
    assert np.array_equal(recovered_a, v_a), "Should recover original vector v_a"

def test_unbind_vectors_convolution():
    """Test unbinding continuous vectors with convolution."""
    # Create continuous unit vectors
    v_a = normalize_vector(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    v_b = normalize_vector(np.array([0.5, 0.4, 0.3, 0.2, 0.1]))
    
    # Bind with convolution
    bound = bind_vectors(v_a, v_b, method="convolution")
    
    # Unbind to recover approximations
    recovered_b = unbind_vectors(bound, v_a, method="convolution")
    recovered_a = unbind_vectors(bound, v_b, method="convolution")
    
    # For continuous vectors, recovery will be approximate
    # Calculate similarity between original and recovered vectors
    sim_b = calculate_similarity(v_b, recovered_b, method="cosine")
    sim_a = calculate_similarity(v_a, recovered_a, method="cosine")
    
    # Similarity should be high but not perfect
    assert sim_b > 0.7, "Recovered vector should be similar to original"
    assert sim_a > 0.7, "Recovered vector should be similar to original"

# Test bundling vectors
def test_bundle_vectors_binary():
    """Test bundling binary vectors."""
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 0, 1])
    v3 = np.array([0, 0, 1, 1, 0])
    
    result = bundle_vectors([v1, v2, v3])
    
    # For binary vectors with equal weights, result is majority vote
    expected = np.array([0, 1, 0, 1, 1])  # Majority vote for each position
    assert np.array_equal(result, expected), "Bundling binary vectors should use majority vote"

def test_bundle_vectors_weighted():
    """Test bundling vectors with explicit weights."""
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 0, 1])
    
    # Weight the second vector more heavily
    result = bundle_vectors([v1, v2], weights=[0.2, 0.8])
    
    # Expected result with weighted majority vote
    expected = np.array([1, 1, 0, 0, 1])  # v2 dominates due to higher weight
    assert np.array_equal(result, expected), "Weighted bundling should reflect specified weights"

def test_bundle_vectors_continuous():
    """Test bundling continuous vectors."""
    v1 = normalize_vector(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    v2 = normalize_vector(np.array([0.5, 0.4, 0.3, 0.2, 0.1]))
    
    result = bundle_vectors([v1, v2])
    
    # Result should be normalized weighted sum
    expected_unnormalized = (v1 + v2) / 2
    expected = normalize_vector(expected_unnormalized)
    assert np.allclose(result, expected), "Bundling continuous vectors should produce normalized weighted sum"

def test_bundle_vectors_empty():
    """Test bundling empty list raises ValueError."""
    with pytest.raises(ValueError):
        bundle_vectors([])

def test_bundle_vectors_different_shapes():
    """Test bundling vectors with different shapes raises ValueError."""
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        bundle_vectors([v1, v2])

def test_bundle_vectors_wrong_weights():
    """Test bundling with mismatched weights raises ValueError."""
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 0, 1])
    with pytest.raises(ValueError):
        bundle_vectors([v1, v2], weights=[0.5])

# Test vector permutation
def test_permute_vector():
    """Test cyclic shifting of vector elements."""
    v = np.array([1, 2, 3, 4, 5])
    
    # Shift right by 2
    result = permute_vector(v, 2)
    expected = np.array([4, 5, 1, 2, 3])
    assert np.array_equal(result, expected), "Positive shift should rotate elements right"
    
    # Shift left by 1
    result = permute_vector(v, -1)
    expected = np.array([2, 3, 4, 5, 1])
    assert np.array_equal(result, expected), "Negative shift should rotate elements left"
    
    # Shift by length should return same vector
    result = permute_vector(v, len(v))
    assert np.array_equal(result, v), "Shifting by length should return the same vector"

# Test similarity calculations
def test_calculate_similarity_cosine():
    """Test cosine similarity for continuous vectors."""
    v1 = normalize_vector(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    v2 = normalize_vector(np.array([0.5, 0.4, 0.3, 0.2, 0.1]))
    
    similarity = calculate_similarity(v1, v2, method="cosine")
    
    # Calculate expected cosine similarity manually
    dot_product = np.dot(v1, v2)
    expected = (dot_product + 1) / 2  # Map from [-1,1] to [0,1]
    
    assert np.isclose(similarity, expected), "Cosine similarity calculation should be correct"
    assert 0 <= similarity <= 1, "Similarity should be between 0 and 1"

def test_calculate_similarity_hamming():
    """Test Hamming similarity for binary vectors."""
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 0, 1])
    
    similarity = calculate_similarity(v1, v2, method="hamming")
    
    # Calculate expected Hamming similarity manually
    # 2 positions differ out of 5, so similarity is 1 - 2/5 = 0.6
    expected = 0.6
    
    assert np.isclose(similarity, expected), "Hamming similarity calculation should be correct"
    assert 0 <= similarity <= 1, "Similarity should be between 0 and 1"

def test_calculate_similarity_auto():
    """Test auto-detection of similarity method."""
    # Binary vectors should use Hamming similarity
    v1 = np.array([0, 1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 0, 1])
    auto_sim = calculate_similarity(v1, v2, method="auto")
    hamming_sim = calculate_similarity(v1, v2, method="hamming")
    assert np.isclose(auto_sim, hamming_sim), "Auto should detect binary vectors and use Hamming similarity"
    
    # Continuous vectors should use cosine similarity
    v3 = normalize_vector(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    v4 = normalize_vector(np.array([0.5, 0.4, 0.3, 0.2, 0.1]))
    auto_sim = calculate_similarity(v3, v4, method="auto")
    cosine_sim = calculate_similarity(v3, v4, method="cosine")
    assert np.isclose(auto_sim, cosine_sim), "Auto should detect continuous vectors and use cosine similarity"

def test_calculate_similarity_identical():
    """Test similarity of identical vectors is 1.0."""
    # Binary vectors
    v1 = np.array([0, 1, 0, 1, 1])
    assert np.isclose(calculate_similarity(v1, v1), 1.0), "Similarity of identical binary vectors should be 1.0"
    
    # Continuous vectors
    v2 = normalize_vector(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert np.isclose(calculate_similarity(v2, v2), 1.0), "Similarity of identical continuous vectors should be 1.0"

def test_calculate_similarity_orthogonal():
    """Test similarity of orthogonal vectors is close to 0.5."""
    # Create orthogonal vectors
    v1 = normalize_vector(np.array([1, 0, 0, 0, 0]))
    v2 = normalize_vector(np.array([0, 1, 0, 0, 0]))
    
    similarity = calculate_similarity(v1, v2, method="cosine")
    
    # For orthogonal vectors, cosine is 0, mapped to 0.5
    assert np.isclose(similarity, 0.5), "Similarity of orthogonal vectors should be 0.5"

# Test FAISS index creation and similarity search
def test_create_hd_index():
    """Test creating different types of FAISS indices."""
    # Test flat index
    index_flat = create_hd_index(100, index_type="flat")
    assert index_flat.d == 100, "Index dimension should match specified value"
    
    # Test invalid dimension
    with pytest.raises(ValueError):
        create_hd_index(0)
    
    # Test invalid index type
    with pytest.raises(ValueError):
        create_hd_index(100, index_type="invalid")

def test_find_similar_vectors():
    """Test finding similar vectors using FAISS index."""
    # Create test vectors
    vectors = {
        "a": normalize_vector(np.array([0.9, 0.1, 0.1, 0.1, 0.1])),
        "b": normalize_vector(np.array([0.1, 0.9, 0.1, 0.1, 0.1])),
        "c": normalize_vector(np.array([0.1, 0.1, 0.9, 0.1, 0.1])),
        "d": normalize_vector(np.array([0.1, 0.1, 0.1, 0.9, 0.1])),
        "e": normalize_vector(np.array([0.1, 0.1, 0.1, 0.1, 0.9]))
    }
    
    # Create query vector most similar to "a"
    query = normalize_vector(np.array([0.8, 0.2, 0.2, 0.2, 0.2]))
    
    # Create index
    index = create_hd_index(5)
    
    # Find similar vectors
    results = find_similar_vectors(index, vectors, query, top_n=3)
    
    # Check result format
    assert len(results) <= 3, "Should return at most top_n results"
    for id, sim in results:
        assert id in vectors, "Result ID should be from the original vectors dictionary"
        assert 0 <= sim <= 1, "Similarity should be between 0 and 1"
    
    # First result should be "a" (most similar to query)
    assert results[0][0] == "a", "Most similar vector should be first"

def test_find_similar_vectors_empty():
    """Test finding similar vectors with empty vectors dictionary."""
    query = normalize_vector(np.array([0.8, 0.2, 0.2, 0.2, 0.2]))
    index = create_hd_index(5)
    results = find_similar_vectors(index, {}, query)
    assert results == [], "Should return empty list for empty vectors dictionary"

# Run the tests (if script is executed directly)
if __name__ == "__main__":
    # You can add code here to run all tests when the script is executed directly
    pass
