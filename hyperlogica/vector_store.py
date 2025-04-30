"""
Vector Store Module

Pure functional implementation of a vector store for high-dimensional vectors in Hyperlogica,
using FAISS for efficient storage and retrieval.
"""

import os
import pickle
import logging
import copy
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple, Union

# Import vector operations
from .vector_operations import normalize_vector, calculate_similarity

# Configure logging
logger = logging.getLogger(__name__)

def create_store(dimension: int, index_type: str = "flat") -> Dict[str, Any]:
    """
    Create a new vector store with FAISS.
    
    Args:
        dimension (int): Dimensionality of vectors to be stored
        index_type (str): Type of FAISS index ("flat", "ivf", or "hnsw")
        
    Returns:
        Dict[str, Any]: A new vector store dictionary
    """
    if dimension <= 0:
        raise ValueError(f"Dimension must be positive, got {dimension}")
    
    if index_type not in ["flat", "ivf", "hnsw"]:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Create appropriate index based on type
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    elif index_type == "ivf":
        # For IVF, we need a quantizer (flat index) and number of centroids
        quantizer = faiss.IndexFlatIP(dimension)
        nlist = 100  # Number of centroids - can be tuned
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        # For an empty index, we can't train it yet
    elif index_type == "hnsw":
        # HNSW index for approximate search
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)  # 32 is M parameter
    
    return {
        "index": index,
        "dimension": dimension,
        "index_type": index_type,
        "items": {},  # Maps identifiers to data (vectors and ACEP representations)
        "item_ids": [],  # Ordered list of identifiers
        "metadata": {
            "created_at": None,  # Will be set when adding the first vector
            "modified_at": None,  # Will be set when modifying the store
            "item_count": 0
        }
    }

def add_vector(store: Dict[str, Any], identifier: str, vector: np.ndarray, 
               acep_representation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a vector to the store, returning a new store instance.
    
    Args:
        store (Dict[str, Any]): Vector store dictionary
        identifier (str): Unique identifier for the vector
        vector (np.ndarray): Vector to add to the store
        acep_representation (Dict[str, Any]): ACEP representation to associate with the vector
        
    Returns:
        Dict[str, Any]: New store with the vector added
    """
    import datetime
    
    # Validate dimensions
    if vector.shape[0] != store["dimension"]:
        raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match store dimension {store['dimension']}")
    
    # Create a deep copy of the store
    new_store = copy.deepcopy(store)
    
    # Normalize vector (ensure unit length for cosine similarity)
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        raise ValueError("Cannot add zero vector to store")
    
    normalized_vector = vector / vector_norm
    
    # Update timestamps
    current_time = datetime.datetime.now().isoformat()
    if new_store["metadata"]["created_at"] is None:
        new_store["metadata"]["created_at"] = current_time
    new_store["metadata"]["modified_at"] = current_time
    
    # Check if this identifier already exists
    if identifier in new_store["items"]:
        # Update the vector and ACEP representation
        index = new_store["item_ids"].index(identifier)
        new_store["items"][identifier] = {
            "vector": normalized_vector,
            "acep": acep_representation
        }
        
        # For most index types, we need to rebuild the index
        # This is inefficient for large stores, but necessary for correctness
        if new_store["index_type"] != "flat":
            vectors = []
            for item_id in new_store["item_ids"]:
                vectors.append(new_store["items"][item_id]["vector"])
            
            new_store["index"].reset()
            if len(vectors) > 0:
                # Don't use train() on empty index
                vectors_array = np.array(vectors).astype('float32')
                if new_store["index_type"] == "ivf" and not new_store["index"].is_trained:
                    # For IVF, we need to train the index first (once)
                    new_store["index"].train(vectors_array)
                new_store["index"].add(vectors_array)
        else:
            # For flat indices, we can rebuild it from scratch (still immutable)
            vectors = []
            for item_id in new_store["item_ids"]:
                vectors.append(new_store["items"][item_id]["vector"])
            
            new_store["index"] = faiss.IndexFlatIP(new_store["dimension"])
            if vectors:
                new_store["index"].add(np.array(vectors).astype('float32'))
    else:
        # Add new vector
        new_store["item_ids"].append(identifier)
        new_store["items"][identifier] = {
            "vector": normalized_vector,
            "acep": acep_representation
        }
        new_store["metadata"]["item_count"] += 1
        
        # Add to index
        new_store["index"].add(np.array([normalized_vector]).astype('float32'))
    
    return new_store

def get_vector(store: Dict[str, Any], identifier: str) -> Dict[str, Any]:
    """
    Retrieve a vector and its ACEP representation by identifier.
    
    Args:
        store (Dict[str, Any]): Vector store dictionary
        identifier (str): Unique identifier for the vector to retrieve
        
    Returns:
        Dict[str, Any]: Dictionary containing the vector and ACEP representation
        
    Raises:
        KeyError: If the identifier does not exist in the store
    """
    if identifier not in store["items"]:
        raise KeyError(f"Identifier not found in store: {identifier}")
    
    item = store["items"][identifier]
    return {
        "identifier": identifier,
        "vector": item["vector"],
        "acep": item["acep"]
    }

def find_similar_vectors(store: Dict[str, Any], query_vector: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Find the most similar vectors to the query vector.
    
    Args:
        store (Dict[str, Any]): Vector store dictionary
        query_vector (np.ndarray): Vector to compare against the store
        top_n (int): Number of most similar vectors to return
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing similar vectors and their ACEP representations
    """
    # Validate dimensions
    if query_vector.shape[0] != store["dimension"]:
        raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match store dimension {store['dimension']}")
    
    # Empty store check
    if len(store["item_ids"]) == 0:
        return []
    
    # Normalize query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("Cannot search with zero vector")
    
    normalized_query = query_vector / query_norm
    
    # Adjust top_n to not exceed number of vectors in store
    adjusted_top_n = min(top_n, len(store["item_ids"]))
    
    # Search for similar vectors
    distances, indices = store["index"].search(np.array([normalized_query]).astype('float32'), adjusted_top_n)
    
    # Process results
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < 0:  # Some indices might be -1 if there aren't enough results
            continue
            
        item_id = store["item_ids"][idx]
        item = store["items"][item_id]
        similarity = float(distances[0][i])  # Convert to Python float for serialization
        
        results.append({
            "identifier": item_id,
            "vector": item["vector"],
            "acep": item["acep"],
            "similarity": similarity
        })
    
    return results

def filter_store_by_type(store: Dict[str, Any], acep_type: str) -> Dict[str, Any]:
    """
    Filter store to only include items of a specific ACEP type.
    
    Args:
        store (Dict[str, Any]): Vector store dictionary
        acep_type (str): ACEP type to filter by (e.g., "conditional_relation", "factual_assertion")
        
    Returns:
        Dict[str, Any]: Filtered copy of the store
    """
    # Create a new store with the same configuration
    filtered_store = create_store(store["dimension"], store["index_type"])
    
    # Add items that match the type
    for item_id in store["item_ids"]:
        item = store["items"][item_id]
        if item["acep"].get("type") == acep_type:
            # Add to filtered store (this will rebuild the index as needed)
            filtered_store = add_vector(filtered_store, item_id, item["vector"], item["acep"])
    
    return filtered_store

def filter_store_by_entity(store: Dict[str, Any], entity_id: str) -> Dict[str, Any]:
    """
    Filter store to only include items related to a specific entity.
    
    Args:
        store (Dict[str, Any]): Vector store dictionary
        entity_id (str): Entity ID to filter by
        
    Returns:
        Dict[str, Any]: Filtered copy of the store
    """
    # Create a new store with the same configuration
    filtered_store = create_store(store["dimension"], store["index_type"])
    
    # Add items that match the entity
    for item_id in store["item_ids"]:
        item = store["items"][item_id]
        if item["acep"].get("attributes", {}).get("entity_id") == entity_id:
            # Add to filtered store (this will rebuild the index as needed)
            filtered_store = add_vector(filtered_store, item_id, item["vector"], item["acep"])
    
    return filtered_store

def save_store(store: Dict[str, Any], path: str) -> bool:
    """
    Save the vector store to disk (functional wrapper around side-effectful operation).
    
    Args:
        store (Dict[str, Any]): Vector store dictionary to save
        path (str): File path where the store should be saved
        
    Returns:
        bool: True if the store was successfully saved, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
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
        logger.error(f"Failed to save vector store: {str(e)}")
        return False

def load_store(path: str) -> Dict[str, Any]:
    """
    Load a vector store from disk.
    
    Args:
        path (str): File path from which to load the store
        
    Returns:
        Dict[str, Any]: The loaded vector store dictionary
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file exists but does not contain a valid vector store
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

def get_all_items(store: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all items from the store.
    
    Args:
        store (Dict[str, Any]): Vector store dictionary
        
    Returns:
        List[Dict[str, Any]]: List of all items in the store
    """
    items = []
    for item_id in store["item_ids"]:
        item = store["items"][item_id]
        items.append({
            "identifier": item_id,
            "vector": item["vector"],
            "acep": item["acep"]
        })
    
    return items
