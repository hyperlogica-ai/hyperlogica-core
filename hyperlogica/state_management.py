"""
State Management Module

Pure functional implementation of state management for Hyperlogica's reasoning system.
All operations create and return new state objects rather than modifying existing ones.
"""

import os
import pickle
import copy
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def create_state(session_id: str) -> Dict[str, Any]:
    """
    Create a new state for a reasoning session.
    
    Args:
        session_id (str): Unique identifier for the session
        
    Returns:
        Dict[str, Any]: New state dictionary
    """
    timestamp = datetime.now().isoformat()
    
    return {
        "session_id": session_id,
        "timestamp": timestamp,
        "concepts": {},  # Maps concept identifiers to concepts
        "relationships": {},  # Maps relationship identifiers to relationships
        "conclusions": {},  # Maps conclusion identifiers to conclusions
        "references": {},  # Maps reference identifiers to referenced entities
        "metadata": {
            "created_at": timestamp,
            "modified_at": timestamp,
            "active_concepts": [],  # Recently active concepts for context management
            "reasoning_traces": []  # Records of reasoning processes
        },
        "ref_counter": 0  # Counter for generating reference IDs
    }

def add_concept_to_state(state: Dict[str, Any], concept: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a concept to the state, returning a new state.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        concept (Dict[str, Any]): Concept to add
        
    Returns:
        Dict[str, Any]: New state with the concept added
        
    Raises:
        ValueError: If the concept is missing required fields
    """
    # Validate required fields
    if "identifier" not in concept:
        raise ValueError("Concept missing required identifier field")
    
    # Create deep copy of state to ensure immutability
    new_state = copy.deepcopy(state)
    identifier = concept["identifier"]
    
    # Set current timestamp
    current_time = datetime.now().isoformat()
    new_state["metadata"]["modified_at"] = current_time
    
    # Add to concepts dictionary
    new_state["concepts"][identifier] = copy.deepcopy(concept)
    
    # Update active concepts (remove if exists, then add to front)
    active_concepts = new_state["metadata"]["active_concepts"]
    if identifier in active_concepts:
        active_concepts.remove(identifier)
    active_concepts.insert(0, identifier)
    
    # Trim list to keep only recent concepts
    max_active = 10  # Maximum number of active concepts to track
    new_state["metadata"]["active_concepts"] = active_concepts[:max_active]
    
    # Add reference
    new_state["ref_counter"] += 1
    reference_id = f"state[{new_state['ref_counter']}]"
    new_state["references"][reference_id] = {
        "type": "concept",
        "identifier": identifier
    }
    
    logger.debug(f"Added concept to state: {identifier}, reference: {reference_id}")
    return new_state

def add_relation_to_state(state: Dict[str, Any], relation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a relationship to the state, returning a new state.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        relation (Dict[str, Any]): Relationship to add
        
    Returns:
        Dict[str, Any]: New state with the relationship added
        
    Raises:
        ValueError: If the relation is missing required fields
        KeyError: If source or target concept doesn't exist in state
    """
    # Validate required fields
    if "identifier" not in relation:
        raise ValueError("Relation missing required identifier field")
    
    if "source" not in relation or "target" not in relation:
        raise ValueError("Relation missing source or target field")
    
    # Create deep copy of state to ensure immutability
    new_state = copy.deepcopy(state)
    identifier = relation["identifier"]
    source_id = relation["source"]
    target_id = relation["target"]
    
    # Verify that source and target exist in state
    if source_id not in new_state["concepts"]:
        raise KeyError(f"Source concept '{source_id}' not found in state")
    
    if target_id not in new_state["concepts"]:
        raise KeyError(f"Target concept '{target_id}' not found in state")
    
    # Set current timestamp
    current_time = datetime.now().isoformat()
    new_state["metadata"]["modified_at"] = current_time
    
    # Add to relationships dictionary
    new_state["relationships"][identifier] = copy.deepcopy(relation)
    
    # Add reference
    new_state["ref_counter"] += 1
    reference_id = f"state[{new_state['ref_counter']}]"
    new_state["references"][reference_id] = {
        "type": "relation",
        "identifier": identifier
    }
    
    logger.debug(f"Added relation to state: {identifier}, reference: {reference_id}")
    return new_state

def add_conclusion_to_state(state: Dict[str, Any], conclusion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a conclusion to the state, returning a new state.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        conclusion (Dict[str, Any]): Conclusion to add
        
    Returns:
        Dict[str, Any]: New state with the conclusion added
        
    Raises:
        ValueError: If the conclusion is missing required fields
    """
    # Validate required fields
    if "identifier" not in conclusion:
        raise ValueError("Conclusion missing required identifier field")
    
    # Create deep copy of state to ensure immutability
    new_state = copy.deepcopy(state)
    identifier = conclusion["identifier"]
    
    # Set current timestamp
    current_time = datetime.now().isoformat()
    new_state["metadata"]["modified_at"] = current_time
    
    # Add to concepts dictionary (conclusions are a type of concept)
    new_state["concepts"][identifier] = copy.deepcopy(conclusion)
    
    # Also add to conclusions dictionary for easier tracking
    new_state["conclusions"][identifier] = copy.deepcopy(conclusion)
    
    # Update active concepts
    active_concepts = new_state["metadata"]["active_concepts"]
    if identifier in active_concepts:
        active_concepts.remove(identifier)
    active_concepts.insert(0, identifier)
    new_state["metadata"]["active_concepts"] = active_concepts[:10]  # Keep only 10 most recent
    
    # Add reference
    new_state["ref_counter"] += 1
    reference_id = f"state[{new_state['ref_counter']}]"
    new_state["references"][reference_id] = {
        "type": "conclusion",
        "identifier": identifier
    }
    
    logger.debug(f"Added conclusion to state: {identifier}, reference: {reference_id}")
    return new_state

def add_reasoning_trace(state: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a reasoning trace to the state, returning a new state.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        trace (Dict[str, Any]): Reasoning trace to add
        
    Returns:
        Dict[str, Any]: New state with the reasoning trace added
    """
    # Create deep copy of state to ensure immutability
    new_state = copy.deepcopy(state)
    
    # Set current timestamp
    current_time = datetime.now().isoformat()
    new_state["metadata"]["modified_at"] = current_time
    
    # Add trace to reasoning_traces list
    new_state["metadata"]["reasoning_traces"].append(copy.deepcopy(trace))
    
    # Add reference if trace has an identifier
    if "identifier" in trace:
        new_state["ref_counter"] += 1
        reference_id = f"state[{new_state['ref_counter']}]"
        new_state["references"][reference_id] = {
            "type": "reasoning_trace",
            "identifier": trace["identifier"]
        }
        logger.debug(f"Added reasoning trace to state: {trace['identifier']}, reference: {reference_id}")
    else:
        logger.debug(f"Added anonymous reasoning trace to state")
    
    return new_state

def resolve_reference(state: Dict[str, Any], reference: str) -> Dict[str, Any]:
    """
    Resolve a reference to a concept, relation, or attribute in the state.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        reference (str): Reference string (e.g., "state[1]" or "state[1].attribute[name]")
        
    Returns:
        Dict[str, Any]: Resolved entity
        
    Raises:
        KeyError: If reference cannot be resolved
        ValueError: If reference has invalid format
    """
    import re
    
    # Parse direct reference: state[n]
    state_ref_pattern = re.compile(r"state\[(\d+)\]$")
    match = state_ref_pattern.match(reference)
    
    if match:
        ref_id = reference
        if ref_id not in state["references"]:
            raise KeyError(f"Reference '{ref_id}' not found in state")
        
        ref = state["references"][ref_id]
        ref_type = ref["type"]
        ref_identifier = ref["identifier"]
        
        # Return the referenced entity
        if ref_type == "concept":
            if ref_identifier not in state["concepts"]:
                raise KeyError(f"Concept '{ref_identifier}' not found in state")
            return state["concepts"][ref_identifier]
        
        elif ref_type == "relation":
            if ref_identifier not in state["relationships"]:
                raise KeyError(f"Relation '{ref_identifier}' not found in state")
            return state["relationships"][ref_identifier]
        
        elif ref_type == "conclusion":
            if ref_identifier not in state["conclusions"]:
                raise KeyError(f"Conclusion '{ref_identifier}' not found in state")
            return state["conclusions"][ref_identifier]
        
        else:
            raise ValueError(f"Unknown reference type: {ref_type}")
    
    # Parse attribute reference: state[n].attribute[name]
    attr_ref_pattern = re.compile(r"state\[(\d+)\]\.attribute\[([^\]]+)\]")
    match = attr_ref_pattern.match(reference)
    
    if match:
        state_ref = f"state[{match.group(1)}]"
        attr_name = match.group(2)
        
        # First resolve the base reference
        entity = resolve_reference(state, state_ref)
        
        # Then get the attribute
        if "attributes" not in entity or attr_name not in entity["attributes"]:
            raise KeyError(f"Attribute '{attr_name}' not found in reference {state_ref}")
        
        return {
            "type": "attribute",
            "parent": state_ref,
            "name": attr_name,
            "value": entity["attributes"][attr_name]
        }
    
    raise ValueError(f"Invalid reference format: {reference}")

def get_active_context(state: Dict[str, Any], max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Get the currently active concepts to maintain context awareness.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        max_items (int): Maximum number of items to return
        
    Returns:
        List[Dict[str, Any]]: List of active concepts
    """
    active_items = []
    
    # First add items from the active_concepts list
    active_concept_ids = state["metadata"]["active_concepts"]
    
    for concept_id in active_concept_ids[:max_items]:
        if concept_id in state["concepts"]:
            active_items.append(state["concepts"][concept_id])
    
    return active_items

def save_state(state: Dict[str, Any], path: str) -> bool:
    """
    Save the state to disk.
    
    Args:
        state (Dict[str, Any]): State to save
        path (str): File path to save to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Prepare state for serialization
        state_copy = copy.deepcopy(state)
        
        # Set save timestamp
        state_copy["metadata"]["saved_at"] = datetime.now().isoformat()
        
        # Convert numpy arrays to lists for serialization
        state_copy = convert_numpy_to_lists(state_copy)
        
        # Save to disk
        with open(path, 'wb') as f:
            pickle.dump(state_copy, f)
        
        logger.info(f"Saved state to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        return False

def load_state(path: str) -> Dict[str, Any]:
    """
    Load the state from disk.
    
    Args:
        path (str): File path to load from
        
    Returns:
        Dict[str, Any]: Loaded state
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid state
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"State file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Convert lists back to numpy arrays
        state = convert_lists_to_numpy(state)
        
        logger.info(f"Loaded state from {path}")
        return state
    except Exception as e:
        raise ValueError(f"Failed to load state: {e}")

def convert_numpy_to_lists(obj: Any) -> Any:
    """
    Convert numpy arrays to lists for serialization.
    
    Args:
        obj (Any): Object to convert
        
    Returns:
        Any: Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    else:
        return obj

def convert_lists_to_numpy(obj: Any) -> Any:
    """
    Convert lists to numpy arrays after deserialization.
    
    Args:
        obj (Any): Object to convert
        
    Returns:
        Any: Converted object
    """
    if isinstance(obj, dict):
        # Special handling for items with vectors
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

def merge_states(primary_state: Dict[str, Any], secondary_state: Dict[str, Any],
                conflict_resolution: str = "primary") -> Dict[str, Any]:
    """
    Merge two states into a new state.
    
    Args:
        primary_state (Dict[str, Any]): Primary state
        secondary_state (Dict[str, Any]): Secondary state
        conflict_resolution (str): How to resolve conflicts ("primary", "secondary", "newer")
        
    Returns:
        Dict[str, Any]: Merged state
        
    Raises:
        ValueError: If conflict_resolution is invalid
    """
    if conflict_resolution not in ["primary", "secondary", "newer"]:
        raise ValueError(f"Invalid conflict resolution strategy: {conflict_resolution}")
    
    # Create a new state with the primary state's session ID
    merged_state = create_state(primary_state["session_id"])
    
    # Add merge metadata
    merged_state["metadata"]["merge_source"] = {
        "primary_session_id": primary_state["session_id"],
        "secondary_session_id": secondary_state["session_id"],
        "merge_strategy": conflict_resolution,
        "merge_time": datetime.now().isoformat()
    }
    
    # Merge concepts
    for concept_id, concept in primary_state["concepts"].items():
        merged_state = add_concept_to_state(merged_state, concept)
    
    for concept_id, concept in secondary_state["concepts"].items():
        if concept_id not in merged_state["concepts"]:
            # No conflict
            merged_state = add_concept_to_state(merged_state, concept)
        elif conflict_resolution == "secondary":
            # Secondary state overrides
            merged_state = add_concept_to_state(merged_state, concept)
        elif conflict_resolution == "newer":
            # Newer timestamp overrides
            primary_ts = merged_state["concepts"][concept_id].get("timestamp", "")
            secondary_ts = concept.get("timestamp", "")
            if secondary_ts > primary_ts:
                merged_state = add_concept_to_state(merged_state, concept)
    
    # Merge relationships
    for relation_id, relation in primary_state["relationships"].items():
        if relation["source"] in merged_state["concepts"] and relation["target"] in merged_state["concepts"]:
            merged_state = add_relation_to_state(merged_state, relation)
    
    for relation_id, relation in secondary_state["relationships"].items():
        if relation["source"] in merged_state["concepts"] and relation["target"] in merged_state["concepts"]:
            if relation_id not in merged_state["relationships"]:
                # No conflict
                merged_state = add_relation_to_state(merged_state, relation)
            elif conflict_resolution == "secondary":
                # Secondary state overrides
                merged_state = add_relation_to_state(merged_state, relation)
            elif conflict_resolution == "newer":
                # Newer timestamp overrides
                primary_ts = merged_state["relationships"][relation_id].get("timestamp", "")
                secondary_ts = relation.get("timestamp", "")
                if secondary_ts > primary_ts:
                    merged_state = add_relation_to_state(merged_state, relation)
    
    # Merge conclusions
    for conclusion_id, conclusion in primary_state.get("conclusions", {}).items():
        if conclusion_id in merged_state["concepts"]:
            merged_state = add_conclusion_to_state(merged_state, conclusion)
    
    for conclusion_id, conclusion in secondary_state.get("conclusions", {}).items():
        if conclusion_id not in merged_state["conclusions"]:
            if conclusion_id in merged_state["concepts"]:
                merged_state = add_conclusion_to_state(merged_state, conclusion)
    
    # Merge reasoning traces
    for trace in primary_state["metadata"].get("reasoning_traces", []):
        merged_state = add_reasoning_trace(merged_state, trace)
    
    for trace in secondary_state["metadata"].get("reasoning_traces", []):
        merged_state = add_reasoning_trace(merged_state, trace)
    
    return merged_state
