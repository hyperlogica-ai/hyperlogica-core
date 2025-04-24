"""
Vector Operations Module

This module implements core hyperdimensional vector operations for the Hyperlogica system.
All functions are implemented in a pure functional style with no side effects or state.
"""

import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Any
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
        vector = np.random.randint(0, 2, dimension)
    elif vector_type == "bipolar":
        # Generate bipolar vector (-1s and 1s)
        vector = 2 * np.random.randint(0, 2, dimension) - 1
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
    
    Args:
        vector_a (np.ndarray): First input vector.
        vector_b (np.ndarray): Second input vector.
        method (str, optional): Binding method to use. Options are "xor" for binary vectors,
                                "multiply" for bipolar vectors, "convolution" for continuous 
                                vectors, or "auto" to infer the best method. Defaults to "auto".
        
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
        result = np.logical_xor(vector_a.astype(bool), vector_b.astype(bool)).astype(int)
    elif method == "multiply":
        # Element-wise multiplication for bipolar vectors
        result = vector_a * vector_b
    elif method == "convolution":
        # Circular convolution for continuous vectors
        result = np.real(np.fft.ifft(np.fft.fft(vector_a) * np.fft.fft(vector_b)))
        # Normalize the result
        result = normalize_vector(result)
    else:
        raise ValueError(f"Unsupported binding method: {method}. Use 'xor', 'multiply', 'convolution', or 'auto'.")
    
    return result

def unbind_vectors(bound_vector: np.ndarray, vector_a: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Unbind to recover vector_b from bound_vector and vector_a.
    
    Args:
        bound_vector (np.ndarray): The bound vector resulting from binding vector_a and vector_b.
        vector_a (np.ndarray): One of the original vectors used in the binding operation.
        method (str, optional): Unbinding method to use, matching the method used for binding.
                                Options are "xor" for binary vectors, "multiply" for bipolar vectors,
                                "convolution" for continuous vectors, or "auto" to infer the best method.
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
            method = "convolution"  # Continuous vectors
    
    # Perform unbinding based on method
    if method == "xor":
        # XOR unbinding for binary vectors (XOR is its own inverse operation)
        result = np.logical_xor(bound_vector.astype(bool), vector_a.astype(bool)).astype(int)
    elif method == "multiply":
        # Element-wise multiplication for bipolar vectors (with bipolar, multiply is its own inverse)
        result = bound_vector * vector_a
    elif method == "convolution":
        # Approximate inverse of circular convolution for continuous vectors
        # Division in the frequency domain is equivalent to deconvolution
        fft_bound = np.fft.fft(bound_vector)
        fft_a = np.fft.fft(vector_a)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        result = np.real(np.fft.ifft(fft_bound / (fft_a + epsilon)))
        # Normalize the result
        result = normalize_vector(result)
    else:
        raise ValueError(f"Unsupported unbinding method: {method}. Use 'xor', 'multiply', 'convolution', or 'auto'.")
    
    return result

def bundle_vectors(vector_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Bundle multiple vectors with optional weighting to create a superposition.
    
    Args:
        vector_list (list): List of np.ndarray vectors to bundle together.
        weights (list, optional): List of weights to apply to each vector. 
                                  If None, equal weights are used. Defaults to None.
        
    Returns:
        np.ndarray: A normalized vector representing the weighted combination of input vectors.
        
    Raises:
        ValueError: If vector_list is empty, vectors have different dimensions,
                    or weights don't match the number of vectors.
    """
    if not vector_list:
        raise ValueError("Cannot bundle an empty list of vectors")
    
    # Check that all vectors have the same shape
    first_shape = vector_list[0].shape
    if not all(v.shape == first_shape for v in vector_list):
        raise ValueError("All vectors must have the same shape")
    
    # Handle weights
    if weights is None:
        # Equal weighting
        weights = [1.0 / len(vector_list)] * len(vector_list)
    elif len(weights) != len(vector_list):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of vectors ({len(vector_list)})")
    
    # Combine vectors using weighted sum
    result = np.zeros_like(vector_list[0], dtype=float)
    for vector, weight in zip(vector_list, weights):
        result += weight * vector.astype(float)
    
    # For binary vectors, threshold the result
    if np.all(np.logical_or(vector_list[0] == 0, vector_list[0] == 1)):
        result = (result > 0.5).astype(int)
    # For bipolar vectors, take the sign
    elif np.all(np.logical_or(vector_list[0] == -1, vector_list[0] == 1)):
        result = np.sign(result).astype(int)
    else:
        # For continuous vectors, normalize
        result = normalize_vector(result)
    
    return result

def permute_vector(vector: np.ndarray, shift: int) -> np.ndarray:
    """
    Apply cyclic shift to vector for encoding order information.
    
    Args:
        vector (np.ndarray): Vector to be permuted.
        shift (int): Number of positions to shift elements. Positive values shift right,
                     negative values shift left.
        
    Returns:
        np.ndarray: Permuted vector with elements cyclically shifted.
    """
    # Ensure shift is within valid range
    shift = shift % len(vector)
    # Perform cyclic shift
    return np.roll(vector, shift)

def calculate_similarity(vector_a: np.ndarray, vector_b: np.ndarray, method: str = "auto") -> float:
    """
    Calculate similarity between two vectors.
    
    Args:
        vector_a (np.ndarray): First vector.
        vector_b (np.ndarray): Second vector.
        method (str, optional): Similarity calculation method. Options are "cosine" for 
                                continuous vectors, "hamming" for binary vectors, "dot" for
                                bipolar vectors, or "auto" to infer the best method.
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
        # Ensure the result is in [-1, 1] range
        cos_sim = max(-1.0, min(1.0, cos_sim))
        # Map from [-1, 1] to [0, 1]
        return (cos_sim + 1) / 2
    
    elif method == "hamming":
        # Hamming similarity for binary vectors (1 - normalized Hamming distance)
        hamming_distance = np.sum(vector_a != vector_b)
        return 1 - (hamming_distance / len(vector_a))
    
    elif method == "dot":
        # Normalized dot product for bipolar vectors
        dot_product = np.dot(vector_a, vector_b) / len(vector_a)
        # Map from [-1, 1] to [0, 1]
        return (dot_product + 1) / 2
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}. Use 'cosine', 'hamming', 'dot', or 'auto'.")

def create_hd_index(dimension: int, index_type: str = "flat") -> Any:
    """
    Create a FAISS index for efficient similarity search in high-dimensional space.
    
    Args:
        dimension (int): Dimensionality of vectors to be indexed.
        index_type (str, optional): Type of FAISS index to create. Options include "flat" for 
                                    exact search, "ivf" for inverted file, or "hnsw" for 
                                    hierarchical navigable small world graph. Defaults to "flat".
        
    Returns:
        Any: A FAISS index configured for the specified dimension and type.
        
    Raises:
        ValueError: If dimension is not positive or index_type is not supported.
    """
    if dimension <= 0:
        raise ValueError("Dimension must be a positive integer")
    
    if index_type == "flat":
        # Flat index for exact search (cosine similarity)
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    elif index_type == "ivf":
        # IVF index for approximate search with inverted file
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, int(np.sqrt(dimension)), faiss.METRIC_INNER_PRODUCT)
        # Need to train this index before use
    elif index_type == "hnsw":
        # HNSW index for efficient approximate search
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(f"Unsupported index type: {index_type}. Use 'flat', 'ivf', or 'hnsw'.")
    
    return index

def find_similar_vectors(index: Any, vectors: Dict[str, np.ndarray], query_vector: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Find the most similar vectors to the query vector using a FAISS index.
    
    Args:
        index (Any): FAISS index containing vectors.
        vectors (Dict[str, np.ndarray]): Dictionary mapping identifiers to vectors.
        query_vector (np.ndarray): Vector to compare against the index.
        top_n (int, optional): Number of most similar vectors to return. Defaults to 10.
        
    Returns:
        List[Tuple[str, float]]: List of tuples containing identifier and similarity score,
                                sorted by decreasing similarity.
        
    Raises:
        ValueError: If query_vector dimension doesn't match the index dimension.
    """
    if not vectors:
        return []
    
    # Normalize query vector
    query_vector = normalize_vector(query_vector.astype(np.float32))
    
    # Reshape query vector to match FAISS expectations
    query_vector = query_vector.reshape(1, -1)
    
    # Get vector identifiers as list to maintain order
    identifiers = list(vectors.keys())
    
    # Convert vectors to numpy array for FAISS
    vector_array = np.vstack([normalize_vector(vectors[id].astype(np.float32)) for id in identifiers])
    
    # Check if index is empty and needs to be populated
    if index.ntotal == 0:
        index.add(vector_array)
    
    # Perform search
    scores, indices = index.search(query_vector, min(top_n, len(vectors)))
    
    # Map indices back to original identifiers and format results
    results = []
    for i, score in zip(indices[0], scores[0]):
        if i >= 0:  # FAISS may return -1 if not enough results
            # Adjust score from inner product to similarity in [0, 1]
            similarity = (score + 1) / 2
            results.append((identifiers[i], similarity))
    
    return results
