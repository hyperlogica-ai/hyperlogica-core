"""
Vector Operations Module

Pure functional implementation of hyperdimensional vector operations for the Hyperlogica system,
with a focus on core ACEP operations: binding, unbinding, bundling, and normalization.
"""

import numpy as np
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def generate_vector(text: str, dimension: int = 10000, vector_type: str = "binary", seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a deterministic vector representation from text.
    
    Args:
        text (str): Text to generate vector for
        dimension (int): Dimensionality of the vector
        vector_type (str): Type of vector to generate ("binary", "bipolar", or "continuous")
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        np.ndarray: Generated vector representation
    """
    if dimension <= 0:
        raise ValueError(f"Dimension must be positive, got {dimension}")
        
    if vector_type not in ["binary", "bipolar", "continuous"]:
        raise ValueError(f"Unsupported vector type: {vector_type}")
    
    # Generate seed from text if not provided
    if seed is None:
        # Create a deterministic hash from the text
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    
    # Seed the random number generator
    np.random.seed(seed)
    
    # Generate the vector based on vector type
    if vector_type == "binary":
        # Generate binary vector (0s and 1s)
        vector = np.random.randint(0, 2, dimension).astype(np.int8)
    elif vector_type == "bipolar":
        # Generate bipolar vector (-1s and 1s)
        vector = (2 * np.random.randint(0, 2, dimension) - 1).astype(np.int8)
    else:  # continuous
        # Generate continuous vector from normal distribution
        vector = np.random.normal(0, 1, dimension)
        # Normalize continuous vectors to unit length
        vector = normalize_vector(vector)
        
    return vector

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector (np.ndarray): Vector to normalize
        
    Returns:
        np.ndarray: Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-10:  # Avoid division by zero
        raise ValueError("Cannot normalize a zero vector")
        
    return vector / norm

def bind_vectors(vector_a: np.ndarray, vector_b: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Bind two vectors to create a new associated vector.
    
    Args:
        vector_a (np.ndarray): First vector
        vector_b (np.ndarray): Second vector
        method (str): Binding method to use ("xor", "multiply", "convolution", or "auto")
        
    Returns:
        np.ndarray: Bound vector
    """
    if vector_a.shape != vector_b.shape:
        raise ValueError(f"Vectors must have same shape: {vector_a.shape} vs {vector_b.shape}")
    
    # Auto-detect method based on vector type
    if method == "auto":
        if np.all(np.logical_or(vector_a == 0, vector_a == 1)) and np.all(np.logical_or(vector_b == 0, vector_b == 1)):
            method = "xor"  # Binary vectors
        elif np.all(np.logical_or(vector_a == -1, vector_a == 1)) and np.all(np.logical_or(vector_b == -1, vector_b == 1)):
            method = "multiply"  # Bipolar vectors
        else:
            method = "convolution"  # Continuous vectors
    
    # Apply binding method
    if method == "xor":
        # XOR binding for binary vectors
        result = np.logical_xor(vector_a.astype(bool), vector_b.astype(bool)).astype(np.int8)
    elif method == "multiply":
        # Element-wise multiplication for bipolar vectors
        result = vector_a * vector_b
    elif method == "convolution":
        # Circular convolution for continuous vectors
        fft_a = np.fft.fft(vector_a)
        fft_b = np.fft.fft(vector_b)
        result = np.fft.ifft(fft_a * fft_b).real
        result = normalize_vector(result)
    else:
        raise ValueError(f"Unsupported binding method: {method}")
    
    return result

def unbind_vectors(bound_vector: np.ndarray, vector_a: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Unbind vectors to approximately recover vector_b from bound_vector and vector_a.
    
    Args:
        bound_vector (np.ndarray): Result of binding vector_a and vector_b
        vector_a (np.ndarray): One of the original vectors
        method (str): Unbinding method to use ("xor", "multiply", "deconvolution", or "auto")
        
    Returns:
        np.ndarray: Approximation of vector_b
    """
    if bound_vector.shape != vector_a.shape:
        raise ValueError(f"Vectors must have same shape: {bound_vector.shape} vs {vector_a.shape}")
    
    # Auto-detect method based on vector type
    if method == "auto":
        if np.all(np.logical_or(bound_vector == 0, bound_vector == 1)) and np.all(np.logical_or(vector_a == 0, vector_a == 1)):
            method = "xor"  # Binary vectors
        elif np.all(np.logical_or(bound_vector == -1, bound_vector == 1)) and np.all(np.logical_or(vector_a == -1, vector_a == 1)):
            method = "multiply"  # Bipolar vectors
        else:
            method = "deconvolution"  # Continuous vectors
    
    # Apply unbinding method
    if method == "xor":
        # XOR unbinding for binary vectors (XOR is its own inverse)
        result = np.logical_xor(bound_vector.astype(bool), vector_a.astype(bool)).astype(np.int8)
    elif method == "multiply":
        # Element-wise multiplication for bipolar vectors (with bipolar, multiply is its own inverse)
        result = bound_vector * vector_a
    elif method == "deconvolution":
        # Circular correlation (approximate inverse of convolution)
        fft_bound = np.fft.fft(bound_vector)
        fft_a_conj = np.conjugate(np.fft.fft(vector_a))
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        fft_a_abs = np.abs(np.fft.fft(vector_a))
        
        # Element-wise division in frequency domain is equivalent to deconvolution
        result = np.fft.ifft(fft_bound * fft_a_conj / (fft_a_abs**2 + epsilon)).real
        result = normalize_vector(result)
    else:
        raise ValueError(f"Unsupported unbinding method: {method}")
    
    return result

def bundle_vectors(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Bundle vectors into a superposition representing their combination.
    
    Args:
        vectors (List[np.ndarray]): Vectors to bundle
        weights (List[float], optional): Weight for each vector
        
    Returns:
        np.ndarray: Bundled vector
    """
    if not vectors:
        raise ValueError("Cannot bundle an empty list of vectors")
    
    # Check that all vectors have the same shape
    first_shape = vectors[0].shape
    if not all(v.shape == first_shape for v in vectors):
        raise ValueError("All vectors must have the same shape")
    
    # Handle weights
    if weights is None:
        # Equal weighting
        weights = [1.0 / len(vectors)] * len(vectors)
    elif len(weights) != len(vectors):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of vectors ({len(vectors)})")
    
    # Determine vector type based on first vector
    vector_type = "continuous"  # Default
    if np.all(np.logical_or(vectors[0] == 0, vectors[0] == 1)):
        vector_type = "binary"
    elif np.all(np.logical_or(vectors[0] == -1, vectors[0] == 1)):
        vector_type = "bipolar"
    
    # Combine vectors using weighted sum
    result = np.zeros_like(vectors[0], dtype=float)
    for vector, weight in zip(vectors, weights):
        result += weight * vector.astype(float)
    
    # Process the result based on vector type
    if vector_type == "binary":
        # Threshold for binary vectors: values > 0.5 become 1, others 0
        result = (result > 0.5).astype(np.int8)
    elif vector_type == "bipolar":
        # Sign function for bipolar vectors: positive values become 1, others -1
        result = np.sign(result).astype(np.int8)
    else:
        # Normalize continuous vectors
        result = normalize_vector(result)
    
    return result

def calculate_similarity(vector_a: np.ndarray, vector_b: np.ndarray, method: str = "auto") -> float:
    """
    Calculate similarity between two vectors.
    
    Args:
        vector_a (np.ndarray): First vector
        vector_b (np.ndarray): Second vector
        method (str): Similarity method ("cosine", "hamming", "dot", or "auto")
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Check dimensions
    if vector_a.shape != vector_b.shape:
        raise ValueError(f"Vectors must have same shape: {vector_a.shape} vs {vector_b.shape}")
    
    # Auto-detect method based on vector type
    if method == "auto":
        if np.all(np.logical_or(vector_a == 0, vector_a == 1)) and np.all(np.logical_or(vector_b == 0, vector_b == 1)):
            method = "hamming"  # Binary vectors
        elif np.all(np.logical_or(vector_a == -1, vector_a == 1)) and np.all(np.logical_or(vector_b == -1, vector_b == 1)):
            method = "dot"  # Bipolar vectors
        else:
            method = "cosine"  # Continuous vectors
    
    # Calculate similarity based on method
    if method == "cosine":
        # Cosine similarity for continuous vectors
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0  # Handle zero vectors
        
        cos_sim = np.dot(vector_a, vector_b) / (norm_a * norm_b)
        # Ensure the result is in [-1, 1] range (numerical stability)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        # Map from [-1, 1] to [0, 1]
        return (cos_sim + 1) / 2
    
    elif method == "hamming":
        # Hamming similarity for binary vectors (1 - normalized Hamming distance)
        hamming_distance = np.sum(vector_a != vector_b)
        return 1 - (hamming_distance / vector_a.shape[0])
    
    elif method == "dot":
        # Normalized dot product for bipolar vectors
        dot_product = np.dot(vector_a, vector_b) / vector_a.shape[0]
        # Map from [-1, 1] to [0, 1]
        return (dot_product + 1) / 2
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")

def create_role_vectors(dimension: int, num_roles: int = 4) -> Dict[str, np.ndarray]:
    """
    Create a set of approximately orthogonal role vectors for ACEP representation.
    
    Args:
        dimension (int): Dimensionality of the vectors
        num_roles (int): Number of role vectors to create
        
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping role names to vectors
    """
    roles = {}
    role_names = [
        "condition", "implication", "concept", "relation", 
        "reference", "state", "value", "fact"
    ]
    
    # Create orthogonal vectors for each role
    for i in range(min(num_roles, len(role_names))):
        role_name = role_names[i]
        # Use a different seed for each role to ensure orthogonality
        seed = int(hashlib.md5(role_name.encode()).hexdigest(), 16) % (2**32)
        roles[role_name] = generate_vector(role_name, dimension, "continuous", seed)
    
    return roles

def create_conditional_representation(condition: Dict[str, Any], implication: Dict[str, Any], 
                                     roles: Dict[str, np.ndarray], 
                                     dimension: int = 10000) -> Dict[str, np.ndarray]:
    """
    Create ACEP representation for a conditional rule using vector operations.
    
    Args:
        condition (Dict[str, Any]): Condition part of the rule
        implication (Dict[str, Any]): Implication part of the rule
        roles (Dict[str, np.ndarray]): Role vectors
        dimension (int): Vector dimension
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with vectors for condition, implication, and combined rule
    """
    # Generate vectors for condition components
    condition_concept_text = condition.get("concept", "")
    condition_relation_text = condition.get("relation", "")
    condition_reference_text = condition.get("reference", "")
    
    condition_concept_vector = generate_vector(condition_concept_text, dimension)
    condition_relation_vector = generate_vector(condition_relation_text, dimension)
    condition_reference_vector = generate_vector(condition_reference_text, dimension)
    
    # CRITICAL IMPROVEMENT: Bind concept and relation directly
    # This preserves orthogonality between opposing relations
    concept_relation_vector = bind_vectors(condition_concept_vector, condition_relation_vector)
    
    # Then bind with reference
    condition_compound_vector = bind_vectors(concept_relation_vector, condition_reference_vector)
    
    # Apply role binding for structural marking
    condition_role = roles.get("condition")
    tagged_condition_vector = bind_vectors(condition_compound_vector, condition_role)
    
    # Generate vectors for implication components
    implication_concept_text = implication.get("concept", "")
    implication_state_text = implication.get("state", "")
    
    implication_concept_vector = generate_vector(implication_concept_text, dimension)
    implication_state_vector = generate_vector(implication_state_text, dimension)
    
    # CRITICAL IMPROVEMENT: Bind concept and state directly
    implication_compound_vector = bind_vectors(implication_concept_vector, implication_state_vector)
    
    # Apply role binding for structural marking
    implication_role = roles.get("implication")
    tagged_implication_vector = bind_vectors(implication_compound_vector, implication_role)
    
    # Bundle to create the final rule vector
    rule_vector = bundle_vectors([tagged_condition_vector, tagged_implication_vector])
    
    return {
        "condition_vector": tagged_condition_vector,
        "raw_condition_vector": condition_compound_vector,
        "implication_vector": tagged_implication_vector,
        "rule_vector": rule_vector,
        "component_vectors": {
            "condition_concept": condition_concept_vector,
            "condition_relation": condition_relation_vector,
            "condition_reference": condition_reference_vector,
            "concept_relation": concept_relation_vector,
            "implication_concept": implication_concept_vector,
            "implication_state": implication_state_vector
        }
    }

def create_fact_representation(fact: Dict[str, Any], roles: Dict[str, np.ndarray], 
                              dimension: int = 10000) -> Dict[str, np.ndarray]:
    """
    Create ACEP representation for a factual assertion using vector operations.
    
    Args:
        fact (Dict[str, Any]): Fact content
        roles (Dict[str, np.ndarray]): Role vectors
        dimension (int): Vector dimension
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with fact vector and component vectors
    """
    # Generate vectors for fact components
    concept_text = fact.get("concept", "")
    relation_text = fact.get("relation", "")
    reference_text = fact.get("reference", "")
    
    concept_vector = generate_vector(concept_text, dimension)
    relation_vector = generate_vector(relation_text, dimension)
    reference_vector = generate_vector(reference_text, dimension)
    
    # CRITICAL IMPROVEMENT: Bind concept and relation directly
    concept_relation_vector = bind_vectors(concept_vector, relation_vector)
    
    # Then bind with reference
    fact_compound_vector = bind_vectors(concept_relation_vector, reference_vector)
    
    # Apply role binding for structural marking
    condition_role = roles.get("condition")
    tagged_fact_vector = bind_vectors(fact_compound_vector, condition_role)
    
    # Get actual and reference values if they exist
    actual_value = fact.get("actual_value")
    reference_value = fact.get("reference_value")
    
    if actual_value is not None:
        actual_text = f"{concept_text}_{actual_value}"
        actual_vector = generate_vector(actual_text, dimension)
    else:
        actual_vector = None
    
    if reference_value is not None:
        reference_value_text = f"{reference_text}_{reference_value}"
        reference_value_vector = generate_vector(reference_value_text, dimension)
    else:
        reference_value_vector = None
    
    return {
        "fact_vector": tagged_fact_vector,  # FIXED: Use "fact_vector" for API compatibility
        "raw_fact_vector": fact_compound_vector,
        "component_vectors": {
            "concept": concept_vector,
            "relation": relation_vector,
            "reference": reference_vector,
            "concept_relation": concept_relation_vector,
            "actual_value": actual_vector,
            "reference_value": reference_value_vector
        }
    }

def match_condition_to_fact(rule: Dict[str, Any], fact: Dict[str, Any], 
                          roles: Dict[str, np.ndarray],
                          similarity_threshold: float = 0.7) -> Tuple[bool, float]:
    """
    Check if a fact matches a rule's condition using vector similarity.
    
    Args:
        rule (Dict[str, Any]): Rule representation with condition vector
        fact (Dict[str, Any]): Fact representation with vector
        roles (Dict[str, np.ndarray]): Role vectors for structure
        similarity_threshold (float): Threshold for considering a match
        
    Returns:
        Tuple[bool, float]: (match_result, similarity_score)
    """
    # Get vectors
    if "condition_vector" not in rule:
        return False, 0.0
    
    # FIXED: Check for "fact_vector" instead of "vector"
    if "fact_vector" not in fact and "vector" not in fact:
        return False, 0.0
    
    condition_vector = rule["condition_vector"]
    # FIXED: Use "fact_vector" if available, fall back to "vector" for compatibility
    fact_vector = fact.get("fact_vector", fact.get("vector"))
    
    # Check for NaN or infinite values
    if np.isnan(condition_vector).any() or np.isinf(condition_vector).any():
        logger.warning("Condition vector contains NaN or Inf values")
        condition_vector = np.nan_to_num(condition_vector)
    
    if np.isnan(fact_vector).any() or np.isinf(fact_vector).any():
        logger.warning("Fact vector contains NaN or Inf values")
        fact_vector = np.nan_to_num(fact_vector)
    
    # Calculate similarity
    similarity = calculate_similarity(condition_vector, fact_vector)
    
    # Get ACEP content for logging
    rule_condition = rule.get("acep", {}).get("content", {}).get("condition", {})
    fact_content = fact.get("acep", {}).get("content", {})
    
    rule_concept = rule_condition.get("concept", "")
    fact_concept = fact_content.get("concept", "")
    
    rule_relation = rule_condition.get("relation", "")
    fact_relation = fact_content.get("relation", "")
    
    rule_reference = rule_condition.get("reference", "")
    fact_reference = fact_content.get("reference", "")
    
    # Check if similarity exceeds threshold
    if similarity >= similarity_threshold:
        logger.info(f"Match found! Similarity: {similarity:.4f}")
        logger.info(f"Rule condition: {rule_concept}-{rule_relation}-{rule_reference}")
        logger.info(f"Fact content: {fact_concept}-{fact_relation}-{fact_reference}")
        return True, similarity
    
    return False, similarity

def cleanse_vector(vector: np.ndarray) -> np.ndarray:
    """
    Clean up a vector by removing NaN/Inf values and normalizing.
    
    Args:
        vector (np.ndarray): Vector to cleanse
        
    Returns:
        np.ndarray: Cleansed vector
    """
    # Replace any NaN/Inf values with zeros
    vector = np.nan_to_num(vector)
    
    # Normalize the vector
    return normalize_vector(vector)

def are_opposite_relations(relation1: str, relation2: str) -> bool:
    """
    Check if two relations are conceptual opposites.
    
    Args:
        relation1 (str): First relation
        relation2 (str): Second relation
        
    Returns:
        bool: True if relations are opposites
    """
    # Define pairs of opposite relations
    opposites = {
        "increasing": ["decreasing", "contracting", "declining", "reducing", "falling"],
        "decreasing": ["increasing", "expanding", "growing", "rising", "accelerating"],
        "expanding": ["contracting", "declining", "decreasing", "reducing"],
        "contracting": ["expanding", "growing", "increasing"],
        "above": ["below", "under", "beneath"],
        "below": ["above", "over", "exceeding"],
        "positive": ["negative", "unfavorable"],
        "negative": ["positive", "favorable"],
        "accelerating": ["slowing", "decelerating"],
        "slowing": ["accelerating", "speeding"],
        "outperforming": ["underperforming"],
        "underperforming": ["outperforming"],
        "buy": ["sell"],
        "sell": ["buy"]
    }
    
    # Normalize to lowercase
    rel1 = relation1.lower()
    rel2 = relation2.lower()
    
    # Check if they're in the opposites dictionary
    if rel1 in opposites and rel2 in opposites.get(rel1, []):
        return True
    
    if rel2 in opposites and rel1 in opposites.get(rel2, []):
        return True
    
    return False

