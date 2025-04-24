"""
State Management Module for Hyperlogica

This module provides pure functions for creating, manipulating, and persisting state
in the Hyperlogica system. The state tracks concepts, relationships, and references
throughout a reasoning session.

Functions follow functional programming principles with no side effects.
"""

import json
import pickle
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import copy
from pathlib import Path


def create_state(session_id: str) -> Dict[str, Any]:
    """
    Create a new state for a session.
    
    Args:
        session_id (str): Unique identifier for the session.
        
    Returns:
        dict: New state dictionary with the following structure:
              {
                  "session_id": str,
                  "timestamp": ISO8601 string,
                  "concepts": {},  # Maps concept identifiers to vectors and metadata
                  "relationships": {},  # Maps relationship identifiers to source, target, and metadata
                  "references": {},  # Maps reference identifiers to resolved entities
                  "metadata": {},  # Session-specific metadata
                  "ref_counter": 0  # Counter for generating reference IDs
              }
    """
    return {
        "session_id": session_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "concepts": {},
        "relationships": {},
        "references": {},
        "metadata": {},
        "ref_counter": 0
    }


def add_concept_to_state(state: Dict[str, Any], concept: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a concept to the state.
    
    Args:
        state (dict): Current state dictionary.
        concept (dict): Concept to add to the state, including identifier, 
                        vector representation, and metadata.
        
    Returns:
        dict: Updated state dictionary with the new concept added.
        
    Raises:
        ValueError: If the concept is missing required fields.
    """
    # Validate the concept
    if "identifier" not in concept:
        raise ValueError("Concept must include an identifier")
    if "vector" not in concept:
        raise ValueError("Concept must include a vector representation")
    
    # Create new state (immutable approach)
    new_state = copy.deepcopy(state)
    
    # Add reference if it doesn't already exist
    if concept["identifier"] not in new_state["concepts"]:
        new_state["ref_counter"] += 1
        ref_id = f"state[{new_state['ref_counter']}]"
        new_state["references"][ref_id] = {
            "type": "concept",
            "identifier": concept["identifier"]
        }
    
    # Add or update the concept
    new_state["concepts"][concept["identifier"]] = {
        "vector": concept["vector"],
        "metadata": concept.get("metadata", {}),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return new_state


def add_relation_to_state(state: Dict[str, Any], relation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a relation to the state.
    
    Args:
        state (dict): Current state dictionary.
        relation (dict): Relation to add to the state, including identifier,
                        source and target concept identifiers, relation type, and metadata.
        
    Returns:
        dict: Updated state dictionary with the new relation added.
        
    Raises:
        ValueError: If the relation is missing required fields.
        KeyError: If source or target concept identifiers don't exist in the state.
    """
    # Validate the relation
    if "identifier" not in relation:
        raise ValueError("Relation must include an identifier")
    if "source" not in relation:
        raise ValueError("Relation must include a source")
    if "target" not in relation:
        raise ValueError("Relation must include a target")
    if "relation_type" not in relation:
        raise ValueError("Relation must include a relation_type")
    
    # Check if source and target exist
    if relation["source"] not in state["concepts"]:
        raise KeyError(f"Source concept '{relation['source']}' not found in state")
    if relation["target"] not in state["concepts"]:
        raise KeyError(f"Target concept '{relation['target']}' not found in state")
    
    # Create new state (immutable approach)
    new_state = copy.deepcopy(state)
    
    # Add reference if it doesn't already exist
    if relation["identifier"] not in new_state["relationships"]:
        new_state["ref_counter"] += 1
        ref_id = f"state[{new_state['ref_counter']}]"
        new_state["references"][ref_id] = {
            "type": "relation",
            "identifier": relation["identifier"]
        }
    
    # Add or update the relation
    new_state["relationships"][relation["identifier"]] = {
        "source": relation["source"],
        "target": relation["target"],
        "relation_type": relation["relation_type"],
        "metadata": relation.get("metadata", {}),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return new_state


def resolve_reference(state: Dict[str, Any], reference: str) -> Dict[str, Any]:
    """
    Resolve a reference to a concept or relation in the state.
    
    Args:
        state (dict): Current state dictionary.
        reference (str): Reference string such as "state[1]" or "state[2].attribute[name]".
        
    Returns:
        dict: Resolved concept, relation, or attribute value.
        
    Raises:
        KeyError: If the reference cannot be resolved within the state.
        ValueError: If the reference syntax is invalid.
    """
    # Check if it's a direct reference to state entry
    if reference in state["references"]:
        ref_data = state["references"][reference]
        ref_type = ref_data["type"]
        ref_id = ref_data["identifier"]
        
        if ref_type == "concept":
            if ref_id not in state["concepts"]:
                raise KeyError(f"Concept '{ref_id}' not found in state")
            return {
                "type": "concept",
                "identifier": ref_id,
                "vector": state["concepts"][ref_id]["vector"],
                "metadata": state["concepts"][ref_id]["metadata"]
            }
        elif ref_type == "relation":
            if ref_id not in state["relationships"]:
                raise KeyError(f"Relation '{ref_id}' not found in state")
            relation = state["relationships"][ref_id]
            return {
                "type": "relation",
                "identifier": ref_id,
                "source": relation["source"],
                "target": relation["target"],
                "relation_type": relation["relation_type"],
                "metadata": relation["metadata"]
            }
        else:
            raise ValueError(f"Unknown reference type: {ref_type}")
    
    # Check if it's an attribute reference
    if "." in reference:
        parts = reference.split(".", 1)
        base_ref = parts[0]
        attribute_part = parts[1]
        
        # Resolve the base reference first
        base_obj = resolve_reference(state, base_ref)
        
        # Parse the attribute part (format should be "attribute[name]")
        if not attribute_part.startswith("attribute[") or not attribute_part.endswith("]"):
            raise ValueError(f"Invalid attribute reference syntax: {attribute_part}")
        
        attr_name = attribute_part[len("attribute["):-1]
        
        # Get the attribute from the metadata
        if "metadata" not in base_obj or attr_name not in base_obj["metadata"]:
            raise KeyError(f"Attribute '{attr_name}' not found in reference {base_ref}")
        
        return {
            "type": "attribute",
            "parent": base_ref,
            "name": attr_name,
            "value": base_obj["metadata"][attr_name]
        }
    
    raise ValueError(f"Invalid reference syntax: {reference}")


def get_active_context(state: Dict[str, Any], max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Get the currently active concepts and relations.
    
    Args:
        state (dict): Current state dictionary.
        max_items (int, optional): Maximum number of items to return. Defaults to 10.
        
    Returns:
        list: List of dictionaries representing the active concepts and relations,
              ordered by recency of use or explicit priority marking.
    """
    items = []
    
    # Get concepts and convert to a common format
    for concept_id, concept_data in state["concepts"].items():
        items.append({
            "type": "concept",
            "identifier": concept_id,
            "timestamp": concept_data["timestamp"],
            "priority": concept_data.get("metadata", {}).get("priority", 0),
            "data": {
                "vector": concept_data["vector"],
                "metadata": concept_data["metadata"]
            }
        })
    
    # Get relationships and convert to a common format
    for relation_id, relation_data in state["relationships"].items():
        items.append({
            "type": "relation",
            "identifier": relation_id,
            "timestamp": relation_data["timestamp"],
            "priority": relation_data.get("metadata", {}).get("priority", 0),
            "data": {
                "source": relation_data["source"],
                "target": relation_data["target"],
                "relation_type": relation_data["relation_type"],
                "metadata": relation_data["metadata"]
            }
        })
    
    # Sort items by priority (descending) and then timestamp (descending)
    sorted_items = sorted(
        items,
        key=lambda x: (-x["priority"], x["timestamp"]),
        reverse=True
    )
    
    # Return the top N items
    return sorted_items[:max_items]


def mark_item_active(state: Dict[str, Any], item_id: str, item_type: str) -> Dict[str, Any]:
    """
    Mark an item as recently active by updating its timestamp.
    
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
    new_state = copy.deepcopy(state)
    
    # Update timestamp based on item type
    if item_type == "concept":
        if item_id not in new_state["concepts"]:
            raise KeyError(f"Concept '{item_id}' not found in state")
        new_state["concepts"][item_id]["timestamp"] = datetime.datetime.now().isoformat()
    else:  # item_type == "relation"
        if item_id not in new_state["relationships"]:
            raise KeyError(f"Relation '{item_id}' not found in state")
        new_state["relationships"][item_id]["timestamp"] = datetime.datetime.now().isoformat()
    
    return new_state


def set_item_priority(state: Dict[str, Any], item_id: str, item_type: str, priority: int) -> Dict[str, Any]:
    """
    Set the priority of an item in the state.
    
    Args:
        state (dict): Current state dictionary.
        item_id (str): Identifier of the item.
        item_type (str): Type of the item ("concept" or "relation").
        priority (int): Priority value (higher values indicate higher priority).
        
    Returns:
        dict: Updated state dictionary with the item's priority set.
        
    Raises:
        ValueError: If item_type is invalid.
        KeyError: If item_id doesn't exist in the state.
    """
    if item_type not in ["concept", "relation"]:
        raise ValueError(f"Invalid item type: {item_type}. Must be 'concept' or 'relation'")
    
    # Create new state (immutable approach)
    new_state = copy.deepcopy(state)
    
    # Update priority based on item type
    if item_type == "concept":
        if item_id not in new_state["concepts"]:
            raise KeyError(f"Concept '{item_id}' not found in state")
        if "metadata" not in new_state["concepts"][item_id]:
            new_state["concepts"][item_id]["metadata"] = {}
        new_state["concepts"][item_id]["metadata"]["priority"] = priority
    else:  # item_type == "relation"
        if item_id not in new_state["relationships"]:
            raise KeyError(f"Relation '{item_id}' not found in state")
        if "metadata" not in new_state["relationships"][item_id]:
            new_state["relationships"][item_id]["metadata"] = {}
        new_state["relationships"][item_id]["metadata"]["priority"] = priority
    
    return new_state


def update_metadata(state: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """
    Update session metadata.
    
    Args:
        state (dict): Current state dictionary.
        key (str): Metadata key to update.
        value (Any): Value to set for the key.
        
    Returns:
        dict: Updated state dictionary with the new metadata.
    """
    # Create new state (immutable approach)
    new_state = copy.deepcopy(state)
    new_state["metadata"][key] = value
    return new_state


def get_concept(state: Dict[str, Any], concept_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a concept by its identifier.
    
    Args:
        state (dict): Current state dictionary.
        concept_id (str): Identifier of the concept to retrieve.
        
    Returns:
        Optional[dict]: The concept if found, None otherwise.
    """
    if concept_id not in state["concepts"]:
        return None
    
    concept_data = state["concepts"][concept_id]
    return {
        "type": "concept",
        "identifier": concept_id,
        "vector": concept_data["vector"],
        "metadata": concept_data.get("metadata", {})
    }


def get_relation(state: Dict[str, Any], relation_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a relation by its identifier.
    
    Args:
        state (dict): Current state dictionary.
        relation_id (str): Identifier of the relation to retrieve.
        
    Returns:
        Optional[dict]: The relation if found, None otherwise.
    """
    if relation_id not in state["relationships"]:
        return None
    
    relation_data = state["relationships"][relation_id]
    return {
        "type": "relation",
        "identifier": relation_id,
        "source": relation_data["source"],
        "target": relation_data["target"],
        "relation_type": relation_data["relation_type"],
        "metadata": relation_data.get("metadata", {})
    }


def get_reference_by_index(state: Dict[str, Any], index: int) -> Optional[str]:
    """
    Get a reference key by its index.
    
    Args:
        state (dict): Current state dictionary.
        index (int): Reference index to retrieve.
        
    Returns:
        Optional[str]: The reference key if found, None otherwise.
    """
    ref_key = f"state[{index}]"
    return ref_key if ref_key in state["references"] else None


def get_all_concepts(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all concepts in the state.
    
    Args:
        state (dict): Current state dictionary.
        
    Returns:
        List[dict]: List of all concepts.
    """
    return [
        {
            "type": "concept",
            "identifier": concept_id,
            "vector": concept_data["vector"],
            "metadata": concept_data.get("metadata", {})
        }
        for concept_id, concept_data in state["concepts"].items()
    ]


def get_all_relations(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all relations in the state.
    
    Args:
        state (dict): Current state dictionary.
        
    Returns:
        List[dict]: List of all relations.
    """
    return [
        {
            "type": "relation",
            "identifier": relation_id,
            "source": relation_data["source"],
            "target": relation_data["target"],
            "relation_type": relation_data["relation_type"],
            "metadata": relation_data.get("metadata", {})
        }
        for relation_id, relation_data in state["relationships"].items()
    ]


def find_relations_by_source(state: Dict[str, Any], source_id: str) -> List[Dict[str, Any]]:
    """
    Find all relations with a specific source.
    
    Args:
        state (dict): Current state dictionary.
        source_id (str): Source concept identifier to search for.
        
    Returns:
        List[dict]: List of relations with the specified source.
    """
    return [
        {
            "type": "relation",
            "identifier": relation_id,
            "source": relation_data["source"],
            "target": relation_data["target"],
            "relation_type": relation_data["relation_type"],
            "metadata": relation_data.get("metadata", {})
        }
        for relation_id, relation_data in state["relationships"].items()
        if relation_data["source"] == source_id
    ]


def find_relations_by_target(state: Dict[str, Any], target_id: str) -> List[Dict[str, Any]]:
    """
    Find all relations with a specific target.
    
    Args:
        state (dict): Current state dictionary.
        target_id (str): Target concept identifier to search for.
        
    Returns:
        List[dict]: List of relations with the specified target.
    """
    return [
        {
            "type": "relation",
            "identifier": relation_id,
            "source": relation_data["source"],
            "target": relation_data["target"],
            "relation_type": relation_data["relation_type"],
            "metadata": relation_data.get("metadata", {})
        }
        for relation_id, relation_data in state["relationships"].items()
        if relation_data["target"] == target_id
    ]


def find_relations_by_type(state: Dict[str, Any], relation_type: str) -> List[Dict[str, Any]]:
    """
    Find all relations of a specific type.
    
    Args:
        state (dict): Current state dictionary.
        relation_type (str): Relation type to search for.
        
    Returns:
        List[dict]: List of relations of the specified type.
    """
    return [
        {
            "type": "relation",
            "identifier": relation_id,
            "source": relation_data["source"],
            "target": relation_data["target"],
            "relation_type": relation_data["relation_type"],
            "metadata": relation_data.get("metadata", {})
        }
        for relation_id, relation_data in state["relationships"].items()
        if relation_data["relation_type"] == relation_type
    ]


def get_state_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of the state.
    
    Args:
        state (dict): Current state dictionary.
        
    Returns:
        dict: Summary information about the state.
    """
    return {
        "session_id": state["session_id"],
        "timestamp": state["timestamp"],
        "num_concepts": len(state["concepts"]),
        "num_relationships": len(state["relationships"]),
        "num_references": len(state["references"]),
        "ref_counter": state["ref_counter"],
        "metadata": state["metadata"]
    }


def remove_concept(state: Dict[str, Any], concept_id: str) -> Dict[str, Any]:
    """
    Remove a concept from the state, along with any relations that reference it.
    
    Args:
        state (dict): Current state dictionary.
        concept_id (str): Identifier of the concept to remove.
        
    Returns:
        dict: Updated state with the concept and related relations removed.
        
    Raises:
        KeyError: If concept_id doesn't exist in the state.
    """
    if concept_id not in state["concepts"]:
        raise KeyError(f"Concept '{concept_id}' not found in state")
    
    # Create new state (immutable approach)
    new_state = copy.deepcopy(state)
    
    # Remove the concept
    del new_state["concepts"][concept_id]
    
    # Remove any relations that reference this concept
    relations_to_remove = []
    for relation_id, relation_data in new_state["relationships"].items():
        if relation_data["source"] == concept_id or relation_data["target"] == concept_id:
            relations_to_remove.append(relation_id)
    
    for relation_id in relations_to_remove:
        del new_state["relationships"][relation_id]
    
    # Remove any references to the removed items
    refs_to_remove = []
    for ref_id, ref_data in new_state["references"].items():
        if ref_data["type"] == "concept" and ref_data["identifier"] == concept_id:
            refs_to_remove.append(ref_id)
        elif ref_data["type"] == "relation" and ref_data["identifier"] in relations_to_remove:
            refs_to_remove.append(ref_id)
    
    for ref_id in refs_to_remove:
        del new_state["references"][ref_id]
    
    return new_state


def remove_relation(state: Dict[str, Any], relation_id: str) -> Dict[str, Any]:
    """
    Remove a relation from the state.
    
    Args:
        state (dict): Current state dictionary.
        relation_id (str): Identifier of the relation to remove.
        
    Returns:
        dict: Updated state with the relation removed.
        
    Raises:
        KeyError: If relation_id doesn't exist in the state.
    """
    if relation_id not in state["relationships"]:
        raise KeyError(f"Relation '{relation_id}' not found in state")
    
    # Create new state (immutable approach)
    new_state = copy.deepcopy(state)
    
    # Remove the relation
    del new_state["relationships"][relation_id]
    
    # Remove any references to the relation
    refs_to_remove = []
    for ref_id, ref_data in new_state["references"].items():
        if ref_data["type"] == "relation" and ref_data["identifier"] == relation_id:
            refs_to_remove.append(ref_id)
    
    for ref_id in refs_to_remove:
        del new_state["references"][ref_id]
    
    return new_state


def save_state(state: Dict[str, Any], path: str, format: str = "pkl") -> bool:
    """
    Save the state to disk.
    
    Args:
        state (dict): State dictionary to save.
        path (str): File path where the state should be saved.
        format (str): Format to save in ("pkl" or "json")
        
    Returns:
        bool: True if the state was successfully saved, False otherwise.
        
    Raises:
        IOError: If the directory doesn't exist or isn't writable.
        ValueError: If format is invalid.
    """
    if format not in ["pkl", "json"]:
        raise ValueError(f"Invalid format: {format}. Must be 'pkl' or 'json'")
    
    # Ensure directory exists
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == "pkl":
            with open(path, "wb") as f:
                pickle.dump(state, f)
        else:  # format == "json"
            # Note: json cannot serialize numpy arrays and some other objects directly
            # This is a simplified version - in production, you'd need custom JSON serialization
            with open(path, "w") as f:
                json.dump(state, f, default=lambda o: str(o))
        return True
    except (IOError, OSError) as e:
        raise IOError(f"Error saving state to {path}: {str(e)}")


def load_state(path: str, format: str = "pkl") -> Dict[str, Any]:
    """
    Load a state from disk.
    
    Args:
        path (str): File path from which to load the state.
        format (str): Format to load from ("pkl" or "json")
        
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
        
        # Validate the state structure
        required_keys = ["session_id", "timestamp", "concepts", "relationships", "references", "metadata"]
        for key in required_keys:
            if key not in state:
                raise ValueError(f"Invalid state file: missing '{key}' key")
        
        return state
    except (pickle.UnpicklingError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid state file format: {str(e)}")


def merge_states(primary_state: Dict[str, Any], secondary_state: Dict[str, Any], 
                conflict_resolution: str = "primary") -> Dict[str, Any]:
    """
    Merge two states together.
    
    Args:
        primary_state (dict): Primary state dictionary.
        secondary_state (dict): Secondary state dictionary to merge in.
        conflict_resolution (str): Strategy for resolving conflicts ("primary", "secondary", 
                                  "newer", or "manual")
        
    Returns:
        dict: New merged state dictionary.
        
    Raises:
        ValueError: If conflict_resolution strategy is invalid.
    """
    if conflict_resolution not in ["primary", "secondary", "newer"]:
        raise ValueError(f"Invalid conflict resolution strategy: {conflict_resolution}")
    
    # Create a new state with the primary state's session ID
    merged_state = create_state(primary_state["session_id"])
    
    # Update metadata
    merged_state["metadata"] = copy.deepcopy(primary_state["metadata"])
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
    
    return merged_state


def diff_states(state1: Dict[str, Any], state2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the difference between two states.
    
    Args:
        state1 (dict): First state dictionary.
        state2 (dict): Second state dictionary to compare against.
        
    Returns:
        dict: Difference summary with concepts and relations added, removed, and modified.
    """
    diff = {
        "concepts": {
            "added": [],
            "removed": [],
            "modified": []
        },
        "relationships": {
            "added": [],
            "removed": [],
            "modified": []
        },
        "metadata": {
            "added": [],
            "removed": [],
            "modified": []
        }
    }
    
    # Compare concepts
    for concept_id in state2["concepts"]:
        if concept_id not in state1["concepts"]:
            diff["concepts"]["added"].append(concept_id)
        elif state1["concepts"][concept_id] != state2["concepts"][concept_id]:
            # Check if it's just a timestamp difference
            concept1 = copy.deepcopy(state1["concepts"][concept_id])
            concept2 = copy.deepcopy(state2["concepts"][concept_id])
            concept1.pop("timestamp", None)
            concept2.pop("timestamp", None)
            
            if concept1 != concept2:
                diff["concepts"]["modified"].append(concept_id)
    
    for concept_id in state1["concepts"]:
        if concept_id not in state2["concepts"]:
            diff["concepts"]["removed"].append(concept_id)
    
    # Compare relationships
    for relation_id in state2["relationships"]:
        if relation_id not in state1["relationships"]:
            diff["relationships"]["added"].append(relation_id)
        elif state1["relationships"][relation_id] != state2["relationships"][relation_id]:
            # Check if it's just a timestamp difference
            relation1 = copy.deepcopy(state1["relationships"][relation_id])
            relation2 = copy.deepcopy(state2["relationships"][relation_id])
            relation1.pop("timestamp", None)
            relation2.pop("timestamp", None)
            
            if relation1 != relation2:
                diff["relationships"]["modified"].append(relation_id)
    
    for relation_id in state1["relationships"]:
        if relation_id not in state2["relationships"]:
            diff["relationships"]["removed"].append(relation_id)
    
    # Compare metadata
    for key in state2["metadata"]:
        if key not in state1["metadata"]:
            diff["metadata"]["added"].append(key)
        elif state1["metadata"][key] != state2["metadata"][key]:
            diff["metadata"]["modified"].append(key)
    
    for key in state1["metadata"]:
        if key not in state2["metadata"]:
            diff["metadata"]["removed"].append(key)
    
    return diff