"""
Vector Operations Module

This module implements core hyperdimensional vector operations for the Hyperlogica system.
All functions are implemented in a pure functional style with no side effects or state.
"""

import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Any, Callable
import hashlib
from scipy import signal
import faiss

def generate_vector(text: str, dimension: int = 10000, vector_type: str = "binary", seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a vector representation from text.
    
    Args:
        text (str): Text to convert to a vector representation.
        dimension (int, optional): Dimensionality of the vector to generate. Defaults to 10000.
        vector_type (str, optional): Type of vector to generate. Options are "binary" for 
                                    binary vectors, "bipolar" for {-1, 1} vectors, or 
                                    "continuous" for real-valued vectors. Defaults to "binary".
        seed (int, optional): Random seed for reproducible vector generation.
                             If None, a hash of the text is used as seed. Defaults to None.
        
    Returns:
        np.ndarray: A vector of the specified dimension representing the input text.
        
    Raises:
        ValueError: If dimension is not positive or vector_type is not supported.
    """
    if dimension <= 0:
        raise ValueError("Dimension must be a positive integer")
        
    if vector_type not in ["binary", "bipolar", "continuous"]:
        raise ValueError("Vector type must be one of: 'binary', 'bipolar', 'continuous'")
    
    # Generate seed from text if not provided
    if seed is None:
        # Create a reproducible hash from the text
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        seed = text_hash
    
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
        vector (np.ndarray): Input vector to normalize.
        
    Returns:
        np.ndarray: Normalized vector with the same direction but unit length.
        
    Raises:
        ValueError: If the input is a zero vector which cannot be normalized.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-10:  # Avoid division by zero or very small numbers
        raise ValueError("Cannot normalize a zero vector")
        
    return vector / norm

def bind_vectors(vector_a: np.ndarray, vector_b: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Bind two vectors using specified method to create a new associated vector.
    According to ACEP specification, binding combines vectors to represent their association,
    with the result being dissimilar to the inputs but containing their combined information.
    
    Args:
        vector_a (np.ndarray): First input vector.
        vector_b (np.ndarray): Second input vector.
        method (str, optional): Binding method to use. Options are:
                                - "xor" for binary vectors (element-wise XOR)
                                - "multiply" for bipolar vectors (element-wise multiplication)
                                - "convolution" for continuous vectors (circular convolution)
                                - "auto" to infer the best method based on vector properties
                                Defaults to "auto".
        
    Returns:
        np.ndarray: A new vector representing the binding of the input vectors.
        
    Raises:
        ValueError: If vectors have different dimensions or if an invalid binding method is specified.
    """
    # Check dimensions
    if vector_a.shape != vector_b.shape:
        raise ValueError(f"Vectors must have the same shape: {vector_a.shape} != {vector_b.shape}")
    
    # Detect vector type if method is auto
    if method == "auto":
        if np.all(np.logical_or(vector_a == 0, vector_a == 1)) and np.all(np.logical_or(vector_b == 0, vector_b == 1)):
            method = "xor"  # Binary vectors
        elif np.all(np.logical_or(vector_a == -1, vector_a == 1)) and np.all(np.logical_or(vector_b == -1, vector_b == 1)):
            method = "multiply"  # Bipolar vectors
        else:
            method = "convolution"  # Continuous vectors
    
    # Perform binding based on method
    if method == "xor":
        # XOR binding for binary vectors
        result = np.logical_xor(vector_a.astype(bool), vector_b.astype(bool)).astype(np.int8)
    elif method == "multiply":
        # Element-wise multiplication for bipolar vectors
        result = vector_a * vector_b
    elif method == "convolution":
        # Circular convolution for continuous vectors using FFT for efficiency
        # FFT(a) * FFT(b) = FFT(a ⊛ b) where ⊛ is circular convolution
        result = np.real(np.fft.ifft(np.fft.fft(vector_a) * np.fft.fft(vector_b)))
        # Normalize the result
        result = normalize_vector(result)
    else:
        raise ValueError(f"Unsupported binding method: {method}. Use 'xor', 'multiply', 'convolution', or 'auto'.")
    
    return result

def unbind_vectors(bound_vector: np.ndarray, vector_a: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Unbind to recover vector_b from bound_vector and vector_a.
    If bound_vector = bind(vector_a, vector_b), then vector_b ≈ unbind(bound_vector, vector_a)
    
    Args:
        bound_vector (np.ndarray): The bound vector resulting from binding vector_a and vector_b.
        vector_a (np.ndarray): One of the original vectors used in the binding operation.
        method (str, optional): Unbinding method to use, matching the method used for binding.
                                Options are "xor" for binary vectors, "multiply" for bipolar vectors,
                                "deconvolution" for continuous vectors, or "auto" to infer the best method.
                                Defaults to "auto".
        
    Returns:
        np.ndarray: Recovered approximation of vector_b.
        
    Raises:
        ValueError: If vectors have different dimensions or if an invalid unbinding method is specified.
    """
    # Check dimensions
    if bound_vector.shape != vector_a.shape:
        raise ValueError(f"Vectors must have the same shape: {bound_vector.shape} != {vector_a.shape}")
    
    # Detect vector type if method is auto
    if method == "auto":
        if np.all(np.logical_or(bound_vector == 0, bound_vector == 1)) and np.all(np.logical_or(vector_a == 0, vector_a == 1)):
            method = "xor"  # Binary vectors
        elif np.all(np.logical_or(bound_vector == -1, bound_vector == 1)) and np.all(np.logical_or(vector_a == -1, vector_a == 1)):
            method = "multiply"  # Bipolar vectors
        else:
            method = "deconvolution"  # Continuous vectors
    
    # Perform unbinding based on method
    if method == "xor":
        # XOR unbinding for binary vectors (XOR is its own inverse operation)
        result = np.logical_xor(bound_vector.astype(bool), vector_a.astype(bool)).astype(np.int8)
    elif method == "multiply":
        # Element-wise multiplication for bipolar vectors (with bipolar, multiply is its own inverse)
        result = bound_vector * vector_a
    elif method == "deconvolution":
        # Approximate inverse of circular convolution for continuous vectors using FFT
        fft_bound = np.fft.fft(bound_vector)
        fft_a = np.fft.fft(vector_a)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        # Element-wise division in frequency domain is equivalent to deconvolution
        result = np.real(np.fft.ifft(fft_bound / (fft_a + epsilon)))
        # Normalize the result
        result = normalize_vector(result)
    else:
        raise ValueError(f"Unsupported unbinding method: {method}. Use 'xor', 'multiply', 'deconvolution', or 'auto'.")
    
    return result

def bundle_vectors(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Bundle multiple vectors to create a superposition that represents a combination.
    According to ACEP, bundling combines vectors so the result is similar to all components.
    
    Args:
        vectors (List[np.ndarray]): List of vectors to bundle.
        weights (List[float], optional): Weight for each vector. If None, equal weights are used.
                                        Defaults to None.
        
    Returns:
        np.ndarray: Bundled vector representing the weighted combination.
        
    Raises:
        ValueError: If vectors is empty, vectors have different dimensions,
                   or weights don't match the number of vectors.
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
    if np.all(np.logical_or(vectors[0] == 0, vectors[0] == 1)):
        vector_type = "binary"
    elif np.all(np.logical_or(vectors[0] == -1, vectors[0] == 1)):
        vector_type = "bipolar"
    else:
        vector_type = "continuous"
    
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

def permute_vector(vector: np.ndarray, shift: int) -> np.ndarray:
    """
    Apply cyclic shift to vector for encoding order information.
    In ACEP, permutation reorders vector elements to represent sequential information.
    
    Args:
        vector (np.ndarray): Vector to permute.
        shift (int): Number of positions to shift elements. Positive values shift right,
                     negative values shift left.
        
    Returns:
        np.ndarray: Permuted vector with elements cyclically shifted.
    """
    # Ensure shift is within valid range
    shift = shift % vector.shape[0]
    
    # Perform cyclic shift
    return np.roll(vector, shift)

def random_permutation(vector: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply a random but reproducible permutation to a vector.
    This is useful for creating dissimilar but related vectors.
    
    Args:
        vector (np.ndarray): Vector to permute.
        seed (int, optional): Random seed for reproducible permutation.
                             If None, a random permutation is generated. Defaults to None.
        
    Returns:
        np.ndarray: Randomly permuted vector.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate permutation indices
    indices = np.random.permutation(vector.shape[0])
    
    # Apply permutation
    return vector[indices]

def inverse_permutation(permuted_vector: np.ndarray, shift: int) -> np.ndarray:
    """
    Inverse of cyclic shift permutation.
    
    Args:
        permuted_vector (np.ndarray): Permuted vector.
        shift (int): The original shift value used for permutation.
        
    Returns:
        np.ndarray: Recovered original vector.
    """
    # Apply inverse shift (negative of original shift)
    return np.roll(permuted_vector, -shift % permuted_vector.shape[0])

def calculate_similarity(vector_a: np.ndarray, vector_b: np.ndarray, method: str = "auto") -> float:
    """
    Calculate similarity between two vectors.
    
    Args:
        vector_a (np.ndarray): First vector.
        vector_b (np.ndarray): Second vector.
        method (str, optional): Similarity calculation method:
                                - "cosine" for continuous vectors
                                - "hamming" for binary vectors
                                - "dot" for bipolar vectors
                                - "auto" to infer the best method
                                Defaults to "auto".
        
    Returns:
        float: Similarity score between 0 and 1, where 1 indicates identical vectors
               and 0 indicates orthogonal/completely different vectors.
        
    Raises:
        ValueError: If vectors have different dimensions or if an invalid similarity method is specified.
    """
    # Check dimensions
    if vector_a.shape != vector_b.shape:
        raise ValueError(f"Vectors must have the same shape: {vector_a.shape} != {vector_b.shape}")
    
    # Detect vector type if method is auto
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
        raise ValueError(f"Unsupported similarity method: {method}. Use 'cosine', 'hamming', 'dot', or 'auto'.")

def generate_associative_memory(keys: List[np.ndarray], values: List[np.ndarray], method: str = "auto") -> np.ndarray:
    """
    Generate a Holographic Reduced Representation (HRR) associative memory.
    This creates a single vector that encodes multiple key-value pairs.
    
    Args:
        keys (List[np.ndarray]): List of key vectors.
        values (List[np.ndarray]): List of value vectors (same length as keys).
        method (str, optional): Binding method to use. Defaults to "auto".
        
    Returns:
        np.ndarray: Single vector representing associative memory of all key-value pairs.
        
    Raises:
        ValueError: If keys and values have different lengths or incompatible dimensions.
    """
    if len(keys) != len(values):
        raise ValueError(f"Number of keys ({len(keys)}) must match number of values ({len(values)})")
    
    if not keys:
        raise ValueError("Cannot create associative memory from empty lists")
    
    # Create pairs by binding each key with its corresponding value
    pairs = [bind_vectors(key, value, method) for key, value in zip(keys, values)]
    
    # Bundle all pairs together
    memory = bundle_vectors(pairs)
    
    return memory

def query_associative_memory(memory: np.ndarray, query_key: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Query a Holographic Reduced Representation (HRR) associative memory.
    Retrieves the value associated with the query key from the memory.
    
    Args:
        memory (np.ndarray): Associative memory vector.
        query_key (np.ndarray): Key to lookup in the memory.
        method (str, optional): Binding method that was used to create the memory.
                               Defaults to "auto".
        
    Returns:
        np.ndarray: Retrieved value vector (approximate).
    """
    # Unbind the query key from the memory to get the associated value
    retrieved_value = unbind_vectors(memory, query_key, method)
    
    return retrieved_value

def create_semantic_pointer(concept: str, dimension: int = 10000, vector_type: str = "continuous") -> np.ndarray:
    """
    Create a semantic pointer vector for a concept, as used in Vector Symbolic Architectures.
    
    Args:
        concept (str): Concept to represent as a semantic pointer.
        dimension (int, optional): Dimensionality of the vector. Defaults to 10000.
        vector_type (str, optional): Type of vector to generate. Defaults to "continuous".
        
    Returns:
        np.ndarray: Semantic pointer vector for the concept.
    """
    return generate_vector(concept, dimension, vector_type)

def bind_role_filler(role: np.ndarray, filler: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Bind a role vector with a filler vector to create a role-filler pair.
    This is a common operation in Vector Symbolic Architectures for structured representations.
    
    Args:
        role (np.ndarray): Role vector (e.g., "subject", "object").
        filler (np.ndarray): Filler vector (e.g., "dog", "ball").
        method (str, optional): Binding method to use. Defaults to "auto".
        
    Returns:
        np.ndarray: Bound role-filler pair.
    """
    return bind_vectors(role, filler, method)

def create_sequence_vector(vectors: List[np.ndarray], shifts: Optional[List[int]] = None) -> np.ndarray:
    """
    Create a sequence representation by applying different permutations to each vector.
    
    Args:
        vectors (List[np.ndarray]): List of vectors to include in the sequence.
        shifts (List[int], optional): List of shift values for each position.
                                     If None, consecutive shifts are used. Defaults to None.
        
    Returns:
        np.ndarray: Vector representing the sequence.
        
    Raises:
        ValueError: If vectors is empty.
    """
    if not vectors:
        raise ValueError("Cannot create sequence from empty list")
    
    # Create default shifts if not provided
    if shifts is None:
        shifts = list(range(len(vectors)))
    
    # Apply permutation to each vector based on its position
    permuted_vectors = [permute_vector(v, s) for v, s in zip(vectors, shifts)]
    
    # Bundle all permuted vectors
    sequence = bundle_vectors(permuted_vectors)
    
    return sequence

def cleanse_vector(vector: np.ndarray, vector_type: str = "auto") -> np.ndarray:
    """
    Cleanse a vector by removing noise and ensuring it conforms to its expected type.
    Useful when vectors have been retrieved from associative memory or after multiple operations.
    
    Args:
        vector (np.ndarray): Vector to cleanse.
        vector_type (str, optional): Target vector type. If "auto", infer from vector.
                                    Defaults to "auto".
        
    Returns:
        np.ndarray: Cleansed vector.
    """
    # Infer vector type if auto
    if vector_type == "auto":
        # Check if mostly binary (0s and 1s)
        zeros_and_ones = np.isclose(vector, 0) | np.isclose(vector, 1)
        if np.mean(zeros_and_ones) > 0.9:
            vector_type = "binary"
            # Check if mostly bipolar (-1s and 1s)
            neg_ones_and_ones = np.isclose(vector, -1) | np.isclose(vector, 1)
        elif np.mean(neg_ones_and_ones) > 0.9:
            vector_type = "bipolar"
        else:
            vector_type = "continuous"
    
    # Apply cleansing based on vector type
    if vector_type == "binary":
        return (vector > 0.5).astype(np.int8)
    elif vector_type == "bipolar":
        return np.sign(vector).astype(np.int8)
    else:  # continuous
        return normalize_vector(vector)
    