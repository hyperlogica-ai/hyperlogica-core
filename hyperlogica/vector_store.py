"""
Vector Store Module

This module provides functions for storing and retrieving high-dimensional vectors
using the FAISS library for efficient similarity search.
"""

import os
import pickle
import logging
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple

def create_store(dimension: int, index_type: str = "flat") -> Dict[str, Any]:
    """
    Create a new vector store with FAISS.
    
    Args:
        dimension (int): Dimensionality of vectors to be stored.
        index_type (str, optional): Type of FAISS index to create. Options include "flat" for 
                                    exact search, "ivf" for inverted file, or "hnsw" for 
                                    hierarchical navigable small world graph. Defaults to "flat".
        
    Returns:
        dict: A dictionary containing the FAISS index and associated metadata, with structure:
              {
                  "index": faiss.Index,
                  "dimension": int,
                  "index_type": str,
                  "concepts": dict,  # Maps identifiers to metadata
                  "concept_ids": list  # Ordered list of identifiers
              }
        
    Raises:
        ValueError: If dimension is not positive or index_type is not supported.
    """
    if dimension <= 0:
        raise ValueError(f"Dimension must be positive, got {dimension}")
    
    if index_type not in ["flat", "ivf", "hnsw"]:
        raise ValueError(f"Unsupported index type: {index_type}, must be one of: flat, ivf, hnsw")
    
    # Create appropriate index based on type
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    elif index_type == "ivf":
        # For IVF, we need a quantizer (flat index) and number of centroids
        quantizer = faiss.IndexFlatIP(dimension)
        nlist = 100  # Number of centroids - can be tuned
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 10  # Number of centroids to visit during search - can be tuned
    elif index_type == "hnsw":
        # HNSW index for approximate search
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)  # 32 is M parameter
    
    return {
        "index": index,
        "dimension": dimension,
        "index_type": index_type,
        "concepts": {},  # Maps identifiers to metadata
        "concept_ids": []  # Ordered list of identifiers
    }

def add_vector(store: Dict[str, Any], identifier: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
    """
    Add a vector to the store with metadata.
    
    Args:
        store (dict): Vector store dictionary returned by create_store.
        identifier (str): Unique identifier for the vector.
        vector (np.ndarray): Vector to add to the store.
        metadata (dict): Additional metadata to associate with the vector.
        
    Returns:
        bool: True if the vector was successfully added, False otherwise.
        
    Raises:
        ValueError: If vector dimension doesn't match the store's dimension.
    """
    # Validate dimensions
    if vector.shape[0] != store["dimension"]:
        raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match store dimension {store['dimension']}")
    
    # Normalize vector (ensure unit length for cosine similarity)
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        raise ValueError("Cannot add zero vector to store")
    
    normalized_vector = vector / vector_norm
    
    # Check if this identifier already exists
    if identifier in store["concepts"]:
        # Update the metadata and vector
        index = store["concept_ids"].index(identifier)
        store["concepts"][identifier] = {
            "vector": normalized_vector,
            "metadata": metadata
        }
        
        # For most index types, we need to rebuild the index
        # This is inefficient for large stores, but necessary for correctness
        if store["index_type"] != "flat":
            vectors = []
            for concept_id in store["concept_ids"]:
                vectors.append(store["concepts"][concept_id]["vector"])
            
            store["index"].reset()
            if len(vectors) > 0:
                store["index"].add(np.array(vectors))
        else:
            # For flat indices, we can just update the vector in place
            store["index"].reset()
            vectors = []
            for concept_id in store["concept_ids"]:
                vectors.append(store["concepts"][concept_id]["vector"])
            store["index"].add(np.array(vectors))
    else:
        # Add new vector
        store["concept_ids"].append(identifier)
        store["concepts"][identifier] = {
            "vector": normalized_vector,
            "metadata": metadata
        }
        store["index"].add(np.array([normalized_vector]))
    
    return True

def get_vector(store: Dict[str, Any], identifier: str) -> Dict[str, Any]:
    """
    Retrieve a vector and its metadata by identifier.
    
    Args:
        store (dict): Vector store dictionary.
        identifier (str): Unique identifier for the vector to retrieve.
        
    Returns:
        dict: Dictionary containing the vector and its metadata, with structure:
              {
                  "identifier": str,
                  "vector": np.ndarray,
                  "metadata": dict
              }
              
    Raises:
        KeyError: If the identifier does not exist in the store.
    """
    if identifier not in store["concepts"]:
        raise KeyError(f"Identifier not found in store: {identifier}")
    
    concept = store["concepts"][identifier]
    return {
        "identifier": identifier,
        "vector": concept["vector"],
        "metadata": concept["metadata"]
    }

def find_similar_vectors(store: Dict[str, Any], query_vector: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Find the most similar vectors to the query vector.
    
    Args:
        store (dict): Vector store dictionary.
        query_vector (np.ndarray): Vector to compare against the store.
        top_n (int, optional): Number of most similar vectors to return. Defaults to 10.
        
    Returns:
        list: List of dictionaries containing the top_n most similar vectors and metadata,
              ordered by decreasing similarity, with structure:
              [
                  {
                      "identifier": str,
                      "vector": np.ndarray,
                      "metadata": dict,
                      "similarity": float
                  },
                  ...
              ]
        
    Raises:
        ValueError: If query_vector dimension doesn't match the store's dimension.
    """
    # Validate dimensions
    if query_vector.shape[0] != store["dimension"]:
        raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match store dimension {store['dimension']}")
    
    # Empty store check
    if len(store["concept_ids"]) == 0:
        return []
    
    # Normalize query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("Cannot search with zero vector")
    
    normalized_query = query_vector / query_norm
    
    # Adjust top_n to not exceed number of vectors in store
    adjusted_top_n = min(top_n, len(store["concept_ids"]))
    
    # Search for similar vectors
    distances, indices = store["index"].search(np.array([normalized_query]), adjusted_top_n)
    
    # Process results
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < 0:  # Some indices might be -1 if there aren't enough results
            continue
            
        concept_id = store["concept_ids"][idx]
        concept = store["concepts"][concept_id]
        similarity = float(distances[0][i])  # Convert to Python float for serialization
        
        results.append({
            "identifier": concept_id,
            "vector": concept["vector"],
            "metadata": concept["metadata"],
            "similarity": similarity
        })
    
    return results

def save_store(store: Dict[str, Any], path: str) -> bool:
    """
    Save the vector store to disk.
    
    Args:
        store (dict): Vector store dictionary to save.
        path (str): File path where the store should be saved.
        
    Returns:
        bool: True if the store was successfully saved, False otherwise.
        
    Raises:
        IOError: If the directory doesn't exist or isn't writable.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # We need special handling for faiss indices which aren't pickle-able
        index = store["index"]
        store_copy = store.copy()
        store_copy["index"] = None
        
        # Save the index separately using faiss.write_index
        index_path = path + ".index"
        faiss.write_index(index, index_path)
        
        # Save the rest of the store
        with open(path, 'wb') as f:
            pickle.dump(store_copy, f)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save vector store: {str(e)}")
        return False

def load_store(path: str) -> Dict[str, Any]:
    """
    Load a vector store from disk.
    
    Args:
        path (str): File path from which to load the store.
        
    Returns:
        dict: The loaded vector store dictionary.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file exists but does not contain a valid vector store.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector store file not found: {path}")
    
    index_path = path + ".index"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector store index file not found: {index_path}")
    
    try:
        # Load the store
        with open(path, 'rb') as f:
            store = pickle.load(f)
        
        # Load the index
        store["index"] = faiss.read_index(index_path)
        
        return store
    except Exception as e:
        raise ValueError(f"Failed to load vector store: {str(e)}")
    