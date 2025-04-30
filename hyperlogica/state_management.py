"""
State Management Module for Hyperlogica

Pure functional implementation of state management for the ACEP-based reasoning system.
All operations create and return new state objects rather than modifying existing ones.
"""

import os
import pickle
import copy
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
        "rules": {},     # Maps rule identifiers to rules
        "facts": {},     # Maps fact identifiers to facts
        "conclusions": {},  # Maps conclusion identifiers to conclusions
        "entities": {},  # Maps entity identifiers to entities
        "metadata": {
            "created_at": timestamp,
            "modified_at": timestamp,
            "reasoning_traces": []  # Records of reasoning processes
        }
    }

def add_rule_to_state(state: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a rule to the state, returning a new state.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        rule (Dict[str, Any]): Rule to add
        
    Returns:
        Dict[str, Any]: New state with the rule added
    """
    # Create deep copy of state to ensure immutability
    new_state = copy.deepcopy(state)
    
    # Extract identifier
    identifier = rule.get("identifier", "")
    if not identifier:
        # Generate a random identifier if none provided
        import uuid
        identifier = f"rule_{uuid.uuid4().hex[:8]}"
        rule["identifier"] = identifier
    
    # Set current timestamp
    current_time = datetime.now().isoformat()
    new_state["metadata"]["modified_at"] = current_time
    
    # Add to rules dictionary
    new_state["rules"][identifier] = copy.deepcopy(rule)
    
    # Also add to concepts for general lookup
    new_state["concepts"][identifier] = copy.deepcopy(rule)
    
    logger.debug(f"Added rule to state: {identifier}")
    return new_state

def add_fact_to_state(state: Dict[str, Any], fact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a fact to the state, returning a new state.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        fact (Dict[str, Any]): Fact to add
        
    Returns:
        Dict[str, Any]: New state with the fact added
    """
    # Create deep copy of state to ensure immutability
    new_state = copy.deepcopy(state)
    
    # Extract identifier
    identifier = fact.get("identifier", "")
    if not identifier:
        # Generate a random identifier if none provided
        import uuid
        identifier = f"fact_{uuid.uuid4().hex[:8]}"
        fact["identifier"] = identifier
    
    # Set current timestamp
    current_time = datetime.now().isoformat()
    new_state["metadata"]["modified_at"] = current_time
    
    # Add to facts dictionary
    new_state["facts"][identifier] = copy.deepcopy(fact)
    
    # Also add to concepts for general lookup
    new_state["concepts"][identifier] = copy.deepcopy(fact)
    
    # Add to entity if entity_id is present
    entity_id = fact.get("attributes", {}).get("entity_id", "")
    if entity_id:
        if entity_id not in new_state["entities"]:
            new_state["entities"][entity_id] = {
                "id": entity_id,
                "facts": []
            }
        new_state["entities"][entity_id]["facts"].append(identifier)
    
    logger.debug(f"Added fact to state: {identifier}")
    return new_state

def add_conclusion_to_state(state: Dict[str, Any], conclusion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a conclusion to the state, returning a new state.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        conclusion (Dict[str, Any]): Conclusion to add
        
    Returns:
        Dict[str, Any]: New state with the conclusion added
    """
    # Create deep copy of state to ensure immutability
    new_state = copy.deepcopy(state)
    
    # Extract identifier
    identifier = conclusion.get("identifier", "")
    if not identifier:
        # Generate a random identifier if none provided
        import uuid
        identifier = f"conclusion_{uuid.uuid4().hex[:8]}"
        conclusion["identifier"] = identifier
    
    # Set current timestamp
    current_time = datetime.now().isoformat()
    new_state["metadata"]["modified_at"] = current_time
    
    # Add to conclusions dictionary
    new_state["conclusions"][identifier] = copy.deepcopy(conclusion)
    
    # Also add to concepts for general lookup
    new_state["concepts"][identifier] = copy.deepcopy(conclusion)
    
    # Add to entity if entity_id is present
    entity_id = conclusion.get("attributes", {}).get("entity_id", "")
    if entity_id:
        if entity_id not in new_state["entities"]:
            new_state["entities"][entity_id] = {
                "id": entity_id,
                "facts": [],
                "conclusions": []
            }
        if "conclusions" not in new_state["entities"][entity_id]:
            new_state["entities"][entity_id]["conclusions"] = []
        new_state["entities"][entity_id]["conclusions"].append(identifier)
    
    logger.debug(f"Added conclusion to state: {identifier}")
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
    
    logger.debug(f"Added reasoning trace to state")
    return new_state

def get_entity_facts(state: Dict[str, Any], entity_id: str) -> List[Dict[str, Any]]:
    """
    Get all facts for a specific entity.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        entity_id (str): Entity ID to get facts for
        
    Returns:
        List[Dict[str, Any]]: List of facts for the entity
    """
    if entity_id not in state["entities"]:
        return []
    
    facts = []
    for fact_id in state["entities"][entity_id].get("facts", []):
        if fact_id in state["facts"]:
            facts.append(state["facts"][fact_id])
    
    return facts

def get_entity_conclusions(state: Dict[str, Any], entity_id: str) -> List[Dict[str, Any]]:
    """
    Get all conclusions for a specific entity.
    
    Args:
        state (Dict[str, Any]): Current state dictionary
        entity_id (str): Entity ID to get conclusions for
        
    Returns:
        List[Dict[str, Any]]: List of conclusions for the entity
    """
    if entity_id not in state["entities"]:
        return []
    
    conclusions = []
    for conclusion_id in state["entities"][entity_id].get("conclusions", []):
        if conclusion_id in state["conclusions"]:
            conclusions.append(state["conclusions"][conclusion_id])
    
    return conclusions

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
