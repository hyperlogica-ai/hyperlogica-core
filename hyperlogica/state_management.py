"""
State Management Module for Hyperlogica

This module provides pure functions for creating, manipulating, and persisting state
in the Hyperlogica system. The state tracks concepts, relationships, references, and
conclusions throughout a reasoning session following ACEP (AI Conceptual Exchange Protocol)
principles.

Functions follow functional programming principles with no side effects.
"""

import os
import json
import pickle
import logging
import copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Configure module logger
logger = logging.getLogger(__name__)

def create_state(session_id: str) -> Dict[str, Any]:
    """
    Create a new state for a session following ACEP principles.
    
    Args:
        session_id (str): Unique identifier for the session.
        
    Returns:
        dict: New state dictionary with the following structure:
              {
                  "session_id": str,
                  "timestamp": ISO8601 string,
                  "concepts": {},  # Maps concept identifiers to vectors and metadata
                  "relationships": {},  # Maps relationship identifiers to source, target, and metadata
                  "conclusions": {},  # Maps conclusion identifiers to conclusion data
                  "references": {},  # Maps reference identifiers to resolved entities
                  "acep_header": {}, # ACEP protocol information 
                  "metadata": {}  # Session-specific metadata
              }
    """
    timestamp = datetime.now().isoformat()
    
    return {
        "session_id": session_id,
        "timestamp": timestamp,
        "concepts": {},
        "relationships": {},
        "conclusions": {},
        "references": {},
        "acep_header": {
            "protocol": "ACEP",
            "version": "1.0",
            "timestamp": timestamp
        },
        "metadata": {
            "active_concepts": [],  # Recently active concepts for context management
            "vector_operations": 0,  # Counter for vector operations performed
            "reasoning_steps": 0,   # Counter for reasoning steps performed
        },
        "ref_counter": 0  # Counter for generating reference IDs
    }

def add_concept_to_state(state: Dict[str, Any], concept: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a concept to the state as an ACEP conceptual vector.
    
    Args:
        state (dict): Current state dictionary.
        concept (dict): Concept to add to the state, including identifier, 
                       vector representation, and metadata.
        
    Returns:
        dict: Updated state dictionary with the new concept added.
        
    Raises:
        ValueError: If the concept is missing required fields.
    """
    # Validate required fields
    identifier = concept.get('identifier')
    if not identifier:
        raise ValueError("Concept missing required identifier field")
    
    if 'vector' not in concept:
        raise ValueError("Concept missing required vector field")
    
    # Make a deep copy to avoid reference issues
    updated_state = copy.deepcopy(state)
    
    # Format the concept according to ACEP structure
    acep_concept = {
        "type": concept.get('type', 'concept'),  # Default to 'concept' if not specified
        "identifier": identifier,
        "vector": concept['vector'],
        "timestamp": datetime.now().isoformat(),
        "attributes": concept.get('attributes', {})
    }
    
    # Ensure attributes has the required fields
    if 'certainty' not in acep_concept['attributes']:
        acep_concept['attributes']['certainty'] = 1.0  # Default high certainty
    
    # Add to state
    updated_state['concepts'][identifier] = acep_concept
    
    # Update active concepts (remove if exists, then add to front)
    active_concepts = updated_state['metadata']['active_concepts']
    if identifier in active_concepts:
        active_concepts.remove(identifier)
    active_concepts.insert(0, identifier)
    
    # Trim list to keep only recent N concepts
    max_active = 10  # Maximum number of active concepts to track
    updated_state['metadata']['active_concepts'] = active_concepts[:max_active]
    
    # Add reference using ACEP format
    updated_state['ref_counter'] += 1
    reference_id = f"state[{updated_state['ref_counter']}]"
    updated_state['references'][reference_id] = {
        'type': 'concept', 
        'identifier': identifier
    }
    
    logger.debug(f"Added concept to state: {identifier}, reference: {reference_id}")
    
    return updated_state

def add_relation_to_state(state: Dict[str, Any], relation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a relation to the state following ACEP relationship format.
    
    Args:
        state (dict): Current state dictionary.
        relation (dict): Relation to add to the state, including identifier,
                        source and target concept identifiers, relation type, and metadata.
        
    Returns:
        dict: Updated state dictionary with the new relation added.
        
    Raises:
        ValueError: If the relation is missing required fields.
    """
    # Validate required fields
    required_fields = ["identifier", "source", "target", "relation_type"]
    for field in required_fields:
        if field not in relation:
            raise ValueError(f"Relation missing required field: {field}")
    
    identifier = relation["identifier"]
    source_id = relation["source"]
    target_id = relation["target"]
    
    # Check that source and target exist in the state
    if source_id not in state["concepts"]:
        raise ValueError(f"Source concept '{source_id}' not found in state")
    
    if target_id not in state["concepts"]:
        raise ValueError(f"Target concept '{target_id}' not found in state")
    
    # Make a deep copy to avoid reference issues
    updated_state = copy.deepcopy(state)
    
    # Format the relation according to ACEP format
    acep_relation = {
        "type": "relation",
        "identifier": identifier,
        "source": source_id,
        "target": target_id,
        "relation_type": relation["relation_type"],
        "timestamp": datetime.now().isoformat(),
        "attributes": relation.get("attributes", {})
    }
    
    # Ensure attributes has the required fields
    if 'certainty' not in acep_relation['attributes']:
        acep_relation['attributes']['certainty'] = 1.0  # Default high certainty
    
    # Add to state
    updated_state['relationships'][identifier] = acep_relation
    
    # Add reference using ACEP format
    updated_state['ref_counter'] += 1
    reference_id = f"state[{updated_state['ref_counter']}]"
    updated_state['references'][reference_id] = {
        'type': 'relation', 
        'identifier': identifier
    }
    
    logger.debug(f"Added relation to state: {identifier}, reference: {reference_id}")
    
    return updated_state

def add_conclusion_to_state(state: Dict[str, Any], conclusion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a conclusion to the state, typically produced by a reasoning step.
    
    Args:
        state (dict): Current state dictionary.
        conclusion (dict): Conclusion to add to the state, including identifier,
                          derived from rules and facts.
        
    Returns:
        dict: Updated state dictionary with the new conclusion added.
        
    Raises:
        ValueError: If the conclusion is missing required fields.
    """
    # Validate required fields
    identifier = conclusion.get('identifier')
    if not identifier:
        raise ValueError("Conclusion missing required identifier field")
    
    # Make a deep copy to avoid reference issues
    updated_state = copy.deepcopy(state)
    
    # Format conclusion according to ACEP format - conclusions are a type of concept
    acep_conclusion = {
        "type": conclusion.get('type', 'concept'),
        "identifier": identifier,
        "vector": conclusion.get('vector'),
        "timestamp": datetime.now().isoformat(),
        "attributes": conclusion.get('attributes', {})
    }
    
    # Ensure attributes has the required fields
    if 'certainty' not in acep_conclusion['attributes']:
        acep_conclusion['attributes']['certainty'] = 1.0  # Default high certainty
    
    # Add to concepts (since conclusions are just concepts derived from reasoning)
    updated_state['concepts'][identifier] = acep_conclusion
    
    # Also track in conclusions collection for easier reference
    updated_state['conclusions'][identifier] = acep_conclusion
    
    # Add to active concepts
    active_concepts = updated_state['metadata']['active_concepts']
    if identifier in active_concepts:
        active_concepts.remove(identifier)
    active_concepts.insert(0, identifier)
    updated_state['metadata']['active_concepts'] = active_concepts[:10]  # Keep only 10 most recent
    
    # Update reasoning step counter
    updated_state['metadata']['reasoning_steps'] += 1
    
    # Add reference using ACEP format
    updated_state['ref_counter'] += 1
    reference_id = f"state[{updated_state['ref_counter']}]"
    updated_state['references'][reference_id] = {
        'type': 'concept', 
        'identifier': identifier
    }
    
    logger.debug(f"Added conclusion to state: {identifier}, reference: {reference_id}")
    
    return updated_state

def resolve_reference(state: Dict[str, Any], reference: str) -> Dict[str, Any]:
    """
    Resolve a reference to a concept, relation, or attribute in the state.
    Implements the ACEP state reference system for efficient information retrieval.
    
    Args:
        state (dict): Current state dictionary.
        reference (str): Reference string such as "state[1]" or "state[2].attribute[name]".
        
    Returns:
        dict: Resolved concept, relation, or attribute value.
        
    Raises:
        KeyError: If the reference cannot be resolved within the state.
        ValueError: If the reference syntax is invalid.
    """
    # Check if it's a direct reference pattern: state[n]
    import re
    state_ref_pattern = r"state\[(\d+)\]"
    state_match = re.match(state_ref_pattern, reference)
    
    if state_match:
        ref_id = f"state[{state_match.group(1)}]"
        if ref_id not in state["references"]:
            raise KeyError(f"Reference '{ref_id}' not found in state")
        
        ref_data = state["references"][ref_id]
        ref_type = ref_data["type"]
        ref_identifier = ref_data["identifier"]
        
        if ref_type == "concept":
            if ref_identifier not in state["concepts"]:
                raise KeyError(f"Concept '{ref_identifier}' not found in state")
            return state["concepts"][ref_identifier]
        
        elif ref_type == "relation":
            if ref_identifier not in state["relationships"]:
                raise KeyError(f"Relation '{ref_identifier}' not found in state")
            return state["relationships"][ref_identifier]
        
        else:
            raise ValueError(f"Unknown reference type: {ref_type}")
    
    # Check if it's an attribute reference pattern: state[n].attribute[name]
    attr_ref_pattern = r"state\[(\d+)\]\.attribute\[([^\]]+)\]"
    attr_match = re.match(attr_ref_pattern, reference)
    
    if attr_match:
        state_ref = f"state[{attr_match.group(1)}]"
        attr_name = attr_match.group(2)
        
        # First resolve the base reference
        base_obj = resolve_reference(state, state_ref)
        
        # Then get the attribute
        if "attributes" not in base_obj or attr_name not in base_obj["attributes"]:
            raise KeyError(f"Attribute '{attr_name}' not found in reference {state_ref}")
        
        return {
            "type": "attribute",
            "parent": state_ref,
            "name": attr_name,
            "value": base_obj["attributes"][attr_name]
        }
    
    raise ValueError(f"Invalid reference syntax: {reference}")

def get_most_recent_reference(state: Dict[str, Any], ref_type: Optional[str] = None) -> str:
    """
    Get the most recently added reference, optionally of a specific type.
    
    Args:
        state (dict): Current state dictionary.
        ref_type (str, optional): Type of reference to get ("concept", "relation").
                                  If None, returns the most recent of any type.
        
    Returns:
        str: The most recent reference identifier.
        
    Raises:
        KeyError: If no matching references are found.
    """
    if not state["references"]:
        raise KeyError("No references in state")
    
    # Get the latest reference counter
    max_counter = state["ref_counter"]
    
    # If no type specified, just return the latest reference
    if ref_type is None:
        return f"state[{max_counter}]"
    
    # Otherwise, search backwards for the most recent reference of the specified type
    for i in range(max_counter, 0, -1):
        ref_id = f"state[{i}]"
        if ref_id in state["references"] and state["references"][ref_id]["type"] == ref_type:
            return ref_id
    
    raise KeyError(f"No references of type '{ref_type}' found in state")

def get_active_context(state: Dict[str, Any], max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Get the currently active concepts and relations to maintain context awareness.
    
    Args:
        state (dict): Current state dictionary.
        max_items (int, optional): Maximum number of items to return. Defaults to 10.
        
    Returns:
        list: List of dictionaries representing the active concepts and relations,
              ordered by recency of use.
    """
    active_items = []
    
    # First add items from the active_concepts list (most recent first)
    active_concept_ids = state["metadata"]["active_concepts"][:max_items]
    for concept_id in active_concept_ids:
        if concept_id in state["concepts"]:
            active_items.append(state["concepts"][concept_id])
    
    # If we haven't reached max_items, add recent relations
    remaining_slots = max_items - len(active_items)
    if remaining_slots > 0:
        # Sort relationships by timestamp, most recent first
        sorted_relations = sorted(
            state["relationships"].values(),
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        # Add the most recent relations until we reach max_items
        for relation in sorted_relations[:remaining_slots]:
            active_items.append(relation)
    
    return active_items

def mark_item_active(state: Dict[str, Any], item_id: str, item_type: str) -> Dict[str, Any]:
    """
    Mark an item as recently active by updating its timestamp and active list.
    
    Args:
        state (dict): Current state dictionary.
        item_id (str): Identifier of the item to mark active.
        item_type (str): Type of the item ("concept" or "relation").
        
    Returns:
        dict: Updated state dictionary with the item marked as active.
        
    Raises:
        ValueError: If item_type is invalid.
        KeyError: If item_id doesn't exist in the state.
    """
    if item_type not in ["concept", "relation"]:
        raise ValueError(f"Invalid item type: {item_type}. Must be 'concept' or 'relation'")
    
    # Create new state (immutable approach)
    updated_state = copy.deepcopy(state)
    
    # Update timestamp based on item type
    if item_type == "concept":
        if item_id not in updated_state["concepts"]:
            raise KeyError(f"Concept '{item_id}' not found in state")
        updated_state["concepts"][item_id]["timestamp"] = datetime.now().isoformat()
        
        # Update active concepts list
        active_concepts = updated_state["metadata"]["active_concepts"]
        if item_id in active_concepts:
            active_concepts.remove(item_id)
        active_concepts.insert(0, item_id)
        updated_state["metadata"]["active_concepts"] = active_concepts[:10]
        
    else:  # item_type == "relation"
        if item_id not in updated_state["relationships"]:
            raise KeyError(f"Relation '{item_id}' not found in state")
        updated_state["relationships"][item_id]["timestamp"] = datetime.now().isoformat()
    
    return updated_state

def find_relations_by_concept(state: Dict[str, Any], concept_id: str, role: str = "any") -> List[Dict[str, Any]]:
    """
    Find all relations involving a specific concept, either as source, target, or both.
    
    Args:
        state (dict): Current state dictionary.
        concept_id (str): Concept identifier to search for.
        role (str, optional): Role of the concept in relations:
                             "source", "target", or "any". Defaults to "any".
        
    Returns:
        list: List of relations involving the concept.
    """
    if role not in ["source", "target", "any"]:
        raise ValueError(f"Invalid role: {role}. Must be 'source', 'target', or 'any'")
    
    relations = []
    
    for relation in state["relationships"].values():
        if (role == "source" and relation["source"] == concept_id) or \
           (role == "target" and relation["target"] == concept_id) or \
           (role == "any" and (relation["source"] == concept_id or relation["target"] == concept_id)):
            relations.append(relation)
    
    return relations

def get_concept_by_similarity(state: Dict[str, Any], query_vector: np.ndarray, 
                             threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """
    Find the most similar concept to a query vector.
    
    Args:
        state (dict): Current state dictionary.
        query_vector (np.ndarray): Vector to compare against.
        threshold (float, optional): Minimum similarity threshold. Defaults to 0.7.
        
    Returns:
        dict or None: The most similar concept, or None if no concept exceeds the threshold.
    """
    from .vector_operations import calculate_similarity
    
    best_similarity = 0.0
    best_concept = None
    
    for concept in state["concepts"].values():
        if "vector" not in concept:
            continue
        
        try:
            similarity = calculate_similarity(query_vector, concept["vector"])
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_concept = concept
        except Exception as e:
            logger.warning(f"Error calculating similarity for concept {concept['identifier']}: {e}")
    
    return best_concept

def add_operation_to_state(state: Dict[str, Any], operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add an ACEP operation to the state, which specifies directives for information processing.
    
    Args:
        state (dict): Current state dictionary.
        operation (dict): Operation to add to the state, including type and parameters.
        
    Returns:
        dict: Updated state dictionary with the new operation added.
        
    Raises:
        ValueError: If the operation is missing required fields.
    """
    # Validate required fields
    if "operation" not in operation:
        raise ValueError("Operation missing required 'operation' field")
    
    if "identifier" not in operation:
        # Generate a default identifier if not provided
        operation["identifier"] = f"operation_{state['ref_counter'] + 1}"
    
    # Create new state (immutable approach)
    updated_state = copy.deepcopy(state)
    
    # Initialize operations collection if it doesn't exist
    if "operations" not in updated_state:
        updated_state["operations"] = {}
    
    # Format the operation according to ACEP
    acep_operation = {
        "type": "operation",
        "identifier": operation["identifier"],
        "operation": operation["operation"],
        "parameters": operation.get("parameters", {}),
        "timestamp": datetime.now().isoformat(),
        "attributes": operation.get("attributes", {})
    }
    
    # Add to state
    updated_state["operations"][operation["identifier"]] = acep_operation
    
    # Add reference
    updated_state["ref_counter"] += 1
    reference_id = f"state[{updated_state['ref_counter']}]"
    updated_state["references"][reference_id] = {
        "type": "operation",
        "identifier": operation["identifier"]
    }
    
    logger.debug(f"Added operation to state: {operation['identifier']}, reference: {reference_id}")
    
    return updated_state

def save_state(state: Dict[str, Any], path: str, format: str = "pkl") -> bool:
    """
    Save the state to disk.
    
    Args:
        state (dict): State dictionary to save.
        path (str): File path where the state should be saved.
        format (str, optional): Save format, either "pkl" or "json". Defaults to "pkl".
        
    Returns:
        bool: True if the state was successfully saved, False otherwise.
        
    Raises:
        IOError: If the directory doesn't exist or isn't writable.
    """
    if format not in ["pkl", "json"]:
        raise ValueError(f"Invalid format: {format}. Must be 'pkl' or 'json'")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Add metadata about save time
        state_copy = copy.deepcopy(state)
        state_copy["metadata"]["last_saved"] = datetime.now().isoformat()
        
        # Prepare state for serialization
        if format == "json":
            # JSON can't handle numpy arrays, so we need to convert them
            serializable_state = prepare_state_for_json(state_copy)
            with open(path, 'w') as f:
                json.dump(serializable_state, f, indent=2)
        else:  # format == "pkl"
            with open(path, 'wb') as f:
                pickle.dump(state_copy, f)
        
        logger.info(f"Saved state to {path} with {len(state['concepts'])} concepts, "
                   f"{len(state.get('relationships', {}))} relationships, "
                   f"{len(state.get('conclusions', {}))} conclusions")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save state: {str(e)}")
        return False

def prepare_state_for_json(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a state for JSON serialization by converting numpy arrays to lists.
    
    Args:
        state (dict): State dictionary to prepare.
        
    Returns:
        dict: JSON-serializable version of the state.
    """
    def convert_numpy_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_arrays(item) for item in obj]
        else:
            return obj
    
    return convert_numpy_arrays(state)

def load_state(path: str, format: str = "pkl") -> Dict[str, Any]:
    """
    Load a state from disk.
    
    Args:
        path (str): File path from which to load the state.
        format (str, optional): Format to load from ("pkl" or "json"). Defaults to "pkl".
        
    Returns:
        dict: The loaded state dictionary.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file exists but does not contain a valid state or if format is invalid.
    """
    if format not in ["pkl", "json"]:
        raise ValueError(f"Invalid format: {format}. Must be 'pkl' or 'json'")
    
    if not Path(path).exists():
        raise FileNotFoundError(f"State file not found: {path}")
    
    try:
        if format == "pkl":
            with open(path, "rb") as f:
                state = pickle.load(f)
        else:  # format == "json"
            with open(path, "r") as f:
                state = json.load(f)
                # Convert JSON lists back to numpy arrays for vectors
                state = restore_numpy_arrays(state)
        
        # Validate the state structure
        required_keys = ["session_id", "timestamp", "concepts", "relationships", "references", "metadata"]
        for key in required_keys:
            if key not in state:
                raise ValueError(f"Invalid state file: missing '{key}' key")
        
        logger.info(f"Loaded state from {path} with {len(state['concepts'])} concepts, "
                   f"{len(state.get('relationships', {}))} relationships, "
                   f"{len(state.get('conclusions', {}))} conclusions")
        
        return state
        
    except (pickle.UnpicklingError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid state file format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to load state: {str(e)}")

def restore_numpy_arrays(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restore numpy arrays from lists in a state loaded from JSON.
    
    Args:
        state (dict): State dictionary with lists that should be numpy arrays.
        
    Returns:
        dict: State with numpy arrays restored.
    """
    def convert_lists_to_numpy(obj):
        if isinstance(obj, dict):
            # Special handling for concept vectors
            if "vector" in obj and isinstance(obj["vector"], list):
                obj["vector"] = np.array(obj["vector"])
            return {k: convert_lists_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Check if this looks like a vector (list of numbers)
            if obj and all(isinstance(x, (int, float)) for x in obj):
                return np.array(obj)
            return [convert_lists_to_numpy(item) for item in obj]
        else:
            return obj
    
    return convert_lists_to_numpy(state)

def merge_states(primary_state: Dict[str, Any], secondary_state: Dict[str, Any], 
                conflict_resolution: str = "primary") -> Dict[str, Any]:
    """
    Merge two states together, following ACEP state merging principles.
    
    Args:
        primary_state (dict): Primary state dictionary.
        secondary_state (dict): Secondary state dictionary to merge in.
        conflict_resolution (str, optional): Strategy for resolving conflicts:
                                            "primary" (keep primary state's values),
                                            "secondary" (use secondary state's values),
                                            "newer" (use the more recent values).
                                            Defaults to "primary".
        
    Returns:
        dict: New merged state dictionary.
        
    Raises:
        ValueError: If conflict_resolution strategy is invalid.
    """
    if conflict_resolution not in ["primary", "secondary", "newer"]:
        raise ValueError(f"Invalid conflict resolution strategy: {conflict_resolution}")
    
    # Create a new state with the primary state's session ID
    merged_state = create_state(primary_state["session_id"])
    
    # Update metadata - add information about merge operation
    merged_state["metadata"] = copy.deepcopy(primary_state["metadata"])
    merged_state["metadata"]["merge_source_sessions"] = [
        primary_state["session_id"],
        secondary_state["session_id"]
    ]
    merged_state["metadata"]["merge_timestamp"] = datetime.now().isoformat()
    merged_state["metadata"]["merge_strategy"] = conflict_resolution
    
    # Add additional metadata from secondary state if not present in primary
    for key, value in secondary_state["metadata"].items():
        if key not in merged_state["metadata"]:
            merged_state["metadata"][key] = value
    
    # Merge concepts
    for concept_id, concept_data in primary_state["concepts"].items():
        merged_state["concepts"][concept_id] = copy.deepcopy(concept_data)
    
    for concept_id, concept_data in secondary_state["concepts"].items():
        if concept_id not in merged_state["concepts"]:
            # No conflict, add the concept
            merged_state["concepts"][concept_id] = copy.deepcopy(concept_data)
        else:
            # Conflict, resolve according to strategy
            if conflict_resolution == "secondary":
                merged_state["concepts"][concept_id] = copy.deepcopy(concept_data)
            elif conflict_resolution == "newer":
                primary_timestamp = merged_state["concepts"][concept_id]["timestamp"]
                secondary_timestamp = concept_data["timestamp"]
                if secondary_timestamp > primary_timestamp:
                    merged_state["concepts"][concept_id] = copy.deepcopy(concept_data)
    
    # Merge relationships
    for relation_id, relation_data in primary_state["relationships"].items():
        # Only add relations if both concepts exist in the merged state
        if (relation_data["source"] in merged_state["concepts"] and 
            relation_data["target"] in merged_state["concepts"]):
            merged_state["relationships"][relation_id] = copy.deepcopy(relation_data)
    
    for relation_id, relation_data in secondary_state["relationships"].items():
        # Only add relations if both concepts exist in the merged state
        if (relation_data["source"] in merged_state["concepts"] and 
            relation_data["target"] in merged_state["concepts"]):
            if relation_id not in merged_state["relationships"]:
                # No conflict, add the relation
                merged_state["relationships"][relation_id] = copy.deepcopy(relation_data)
            else:
                # Conflict, resolve according to strategy
                if conflict_resolution == "secondary":
                    merged_state["relationships"][relation_id] = copy.deepcopy(relation_data)
                elif conflict_resolution == "newer":
                    primary_timestamp = merged_state["relationships"][relation_id]["timestamp"]
                    secondary_timestamp = relation_data["timestamp"]
                    if secondary_timestamp > primary_timestamp:
                        merged_state["relationships"][relation_id] = copy.deepcopy(relation_data)
    
    # Merge conclusions (which are a subset of concepts)
    for conclusion_id, conclusion_data in primary_state.get("conclusions", {}).items():
        if conclusion_id in merged_state["concepts"]:
            merged_state["conclusions"][conclusion_id] = merged_state["concepts"][conclusion_id]
    
    for conclusion_id, conclusion_data in secondary_state.get("conclusions", {}).items():
        if conclusion_id in merged_state["concepts"]:
            merged_state["conclusions"][conclusion_id] = merged_state["concepts"][conclusion_id]
    
    # Rebuild references
    merged_state["ref_counter"] = 0
    merged_state["references"] = {}
    
    # Add references for concepts
    for concept_id in merged_state["concepts"]:
        merged_state["ref_counter"] += 1
        ref_id = f"state[{merged_state['ref_counter']}]"
        merged_state["references"][ref_id] = {
            "type": "concept",
            "identifier": concept_id
        }
    
    # Add references for relations
    for relation_id in merged_state["relationships"]:
        merged_state["ref_counter"] += 1
        ref_id = f"state[{merged_state['ref_counter']}]"
        merged_state["references"][ref_id] = {
            "type": "relation",
            "identifier": relation_id
        }
    
    # Merge operations if present
    if "operations" in primary_state or "operations" in secondary_state:
        merged_state["operations"] = {}
        
        # Add operations from primary state
        for op_id, op_data in primary_state.get("operations", {}).items():
            merged_state["operations"][op_id] = copy.deepcopy(op_data)
            
            # Add reference for operation
            merged_state["ref_counter"] += 1
            ref_id = f"state[{merged_state['ref_counter']}]"
            merged_state["references"][ref_id] = {
                "type": "operation",
                "identifier": op_id
            }
        
        # Add non-conflicting operations from secondary state
        for op_id, op_data in secondary_state.get("operations", {}).items():
            if op_id not in merged_state["operations"]:
                merged_state["operations"][op_id] = copy.deepcopy(op_data)
                
                # Add reference for operation
                merged_state["ref_counter"] += 1
                ref_id = f"state[{merged_state['ref_counter']}]"
                merged_state["references"][ref_id] = {
                    "type": "operation",
                    "identifier": op_id
                }
    
    return merged_state