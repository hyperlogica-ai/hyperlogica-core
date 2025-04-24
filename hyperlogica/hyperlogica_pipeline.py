"""
Hyperlogica Processing Pipeline

This module contains the core processing pipeline functions for the Hyperlogica system,
implemented using functional programming principles. This focuses on the main pipeline
that orchestrates the flow of data through the system, from input parsing through
reasoning to output generation.

The pipeline is designed to be composed of pure functions that transform data without
side effects, making the system easier to test, debug, and extend.
"""

import json
import os
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import faiss
import yaml
import concurrent.futures
from functools import partial, reduce
import copy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('hyperlogica')

# Type aliases for improved readability
ConfigDict = Dict[str, Any]
EntityDict = Dict[str, Any]
ResultDict = Dict[str, Any]
RuleDict = Dict[str, Any] 
FactDict = Dict[str, Any]
StoreDict = Dict[str, Any]
StateDict = Dict[str, Any]
VectorDict = Dict[str, Any]

# Result type for functions that might fail
Result = Tuple[Optional[Any], Optional[str]]


def success(value: Any) -> Result:
    """Create a success result with a value and no error."""
    return (value, None)


def error(err_msg: str) -> Result:
    """Create an error result with an error message and no value."""
    return (None, err_msg)


def is_success(result: Result) -> bool:
    """Check if a result is successful (has no error)."""
    return result[1] is None


def is_error(result: Result) -> bool:
    """Check if a result is an error (has an error message)."""
    return result[1] is not None


def get_value(result: Result) -> Any:
    """Get the value from a successful result."""
    if is_error(result):
        raise ValueError(f"Cannot get value from error result: {result[1]}")
    return result[0]


def get_error(result: Result) -> str:
    """Get the error from an error result."""
    if is_success(result):
        raise ValueError("Cannot get error from success result")
    return result[1]


def map_success(result: Result, fn: Callable[[Any], Any]) -> Result:
    """Apply a function to the value if the result is successful."""
    if is_success(result):
        return success(fn(get_value(result)))
    return result  # Return the error unchanged


def handle_error(result: Result, handler: Callable[[str], Any]) -> Any:
    """Handle the error if the result is an error, otherwise return the value."""
    if is_success(result):
        return get_value(result)
    return handler(get_error(result))


def parse_input_config(input_path: str) -> Result:
    """
    Parse input JSON or YAML configuration file.
    
    Args:
        input_path (str): Path to the configuration file.
        
    Returns:
        Result: Success with configuration dictionary or error with message.
    """
    try:
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if not os.path.exists(input_path):
            return error(f"Configuration file not found: {input_path}")
        
        with open(input_path, 'r') as file:
            if file_ext == '.json':
                config = json.load(file)
            elif file_ext in ['.yaml', '.yml']:
                config = yaml.safe_load(file)
            else:
                return error(f"Unsupported configuration file format: {file_ext}")
                
        logger.info(f"Successfully parsed configuration from {input_path}")
        return success(config)
    except json.JSONDecodeError as e:
        return error(f"JSON parsing error: {str(e)}")
    except yaml.YAMLError as e:
        return error(f"YAML parsing error: {str(e)}")
    except Exception as e:
        return error(f"Unexpected error parsing configuration: {str(e)}")


def validate_config(config: Dict[str, Any]) -> Result:
    """
    Validate configuration dictionary and provide defaults for missing values.
    
    Args:
        config (dict): Raw configuration dictionary from parsed JSON/YAML.
        
    Returns:
        Result: Success with validated configuration or error with message.
    """
    # Create a deep copy to avoid modifying the input
    validated = copy.deepcopy(config)
    
    # Check required sections
    required_sections = ["processing", "input_data", "output_schema"]
    for section in required_sections:
        if section not in validated:
            return error(f"Missing required configuration section: {section}")
    
    # Set defaults for processing options
    if "processing" in validated:
        processing = validated["processing"]
        processing.setdefault("vector_dimension", 10000)
        processing.setdefault("vector_type", "binary")
        processing.setdefault("reasoning_approach", "majority")
        processing.setdefault("certainty_propagation", "min")
        processing.setdefault("recalibration_enabled", True)
        processing.setdefault("max_reasoning_depth", 10)
        processing.setdefault("domain_config", {})
    
    # Set defaults for persistence options
    if "persistence" not in validated:
        validated["persistence"] = {}
    persistence = validated["persistence"]
    persistence.setdefault("load_previous_state", False)
    persistence.setdefault("save_state", False)
    persistence.setdefault("previous_state_path", "")
    persistence.setdefault("state_save_path", "")
    
    # Set defaults for logging options
    if "logging" not in validated:
        validated["logging"] = {}
    logging_opts = validated["logging"]
    logging_opts.setdefault("log_level", "info")
    logging_opts.setdefault("log_path", "")
    logging_opts.setdefault("include_vector_operations", False)
    logging_opts.setdefault("include_llm_interactions", True)
    logging_opts.setdefault("include_reasoning_steps", True)
    
    # Set defaults for LLM options
    if "llm" not in validated:
        validated["llm"] = {}
    llm = validated["llm"]
    llm.setdefault("model", "gpt-4")
    llm.setdefault("temperature", 0.0)
    llm.setdefault("max_tokens", 2000)
    
    # Check input data structure
    input_data = validated["input_data"]
    if "rules" not in input_data or not isinstance(input_data["rules"], list):
        return error("Missing or invalid 'rules' in input_data")
    if "entities" not in input_data or not isinstance(input_data["entities"], list):
        return error("Missing or invalid 'entities' in input_data")
    
    # Check output schema structure
    output_schema = validated["output_schema"]
    if "fields" not in output_schema or not isinstance(output_schema["fields"], list):
        return error("Missing or invalid 'fields' in output_schema")
    output_schema.setdefault("format", "json")
    output_schema.setdefault("include_reasoning_trace", False)
    output_schema.setdefault("include_vector_details", False)
    
    logger.info("Configuration validated successfully")
    return success(validated)


def extract_processing_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract processing-related options from configuration.
    
    Args:
        config (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing only processing-related options.
    """
    processing = config.get("processing", {})
    return {
        "vector_dimension": processing.get("vector_dimension", 10000),
        "vector_type": processing.get("vector_type", "binary"),
        "reasoning_approach": processing.get("reasoning_approach", "majority"),
        "certainty_propagation": processing.get("certainty_propagation", "min"),
        "recalibration_enabled": processing.get("recalibration_enabled", True),
        "max_reasoning_depth": processing.get("max_reasoning_depth", 10),
        "domain_config": processing.get("domain_config", {})
    }


def extract_persistence_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract persistence-related options from configuration.
    
    Args:
        config (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing persistence options.
    """
    persistence = config.get("persistence", {})
    return {
        "load_previous_state": persistence.get("load_previous_state", False),
        "previous_state_path": persistence.get("previous_state_path", ""),
        "save_state": persistence.get("save_state", False),
        "state_save_path": persistence.get("state_save_path", "")
    }


def extract_llm_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract LLM-related options from configuration.
    
    Args:
        config (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing LLM options.
    """
    llm = config.get("llm", {})
    return {
        "model": llm.get("model", "gpt-4"),
        "temperature": llm.get("temperature", 0.0),
        "max_tokens": llm.get("max_tokens", 2000)
    }


def extract_logging_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract logging-related options from configuration.
    
    Args:
        config (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing logging options.
    """
    logging_opts = config.get("logging", {})
    return {
        "log_level": logging_opts.get("log_level", "info"),
        "log_path": logging_opts.get("log_path", ""),
        "include_vector_operations": logging_opts.get("include_vector_operations", False),
        "include_llm_interactions": logging_opts.get("include_llm_interactions", True),
        "include_reasoning_steps": logging_opts.get("include_reasoning_steps", True)
    }


def extract_output_schema(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract output schema definition from configuration.
    
    Args:
        config (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing output schema definition.
    """
    output_schema = config.get("output_schema", {})
    return {
        "format": output_schema.get("format", "json"),
        "fields": output_schema.get("fields", []),
        "include_reasoning_trace": output_schema.get("include_reasoning_trace", False),
        "include_vector_details": output_schema.get("include_vector_details", False)
    }


def create_vector_store(dimension: int, index_type: str = "flat") -> StoreDict:
    """
    Create a new vector store with FAISS.
    
    Args:
        dimension (int): Dimensionality of vectors to be stored.
        index_type (str): Type of FAISS index to create.
        
    Returns:
        dict: A dictionary containing the FAISS index and associated metadata.
    """
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    elif index_type == "ivf":
        # Inverted file index for faster search
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 centroids
        index.train(np.random.random((1000, dimension)).astype('float32'))
    elif index_type == "hnsw":
        # Hierarchical navigable small world graph
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node
    else:
        index = faiss.IndexFlatIP(dimension)  # Default to flat index
        
    return {
        "index": index,
        "dimension": dimension,
        "index_type": index_type,
        "concepts": {},  # Maps identifiers to metadata
        "concept_ids": []  # Ordered list of identifiers
    }


def add_vector_to_store(store: StoreDict, identifier: str, vector: np.ndarray, metadata: Dict[str, Any]) -> StoreDict:
    """
    Add a vector to the store with metadata.
    
    This function follows the immutable data principle by creating a new copy of the store
    rather than modifying the input store directly.
    
    Args:
        store (dict): Vector store dictionary.
        identifier (str): Unique identifier for the vector.
        vector (np.ndarray): Vector to add to the store.
        metadata (dict): Additional metadata to associate with the vector.
        
    Returns:
        dict: Updated vector store with the new vector added.
    """
    # Create a copy of the store to avoid modifying the input
    updated_store = copy.deepcopy(store)
    
    # Check vector dimension
    if vector.shape[0] != updated_store["dimension"]:
        logger.error(f"Vector dimension mismatch: expected {updated_store['dimension']}, got {vector.shape[0]}")
        return updated_store
    
    # Normalize the vector to unit length
    normalized_vector = vector / np.linalg.norm(vector)
    
    # Check if vector already exists
    if identifier in updated_store["concepts"]:
        # Update existing vector
        idx = updated_store["concept_ids"].index(identifier)
        
        # Need to rebuild the index when updating
        updated_store["index"].reset()
        updated_store["concepts"][identifier] = {
            "vector": normalized_vector,
            "metadata": metadata
        }
        
        # Re-add all vectors to the index
        all_vectors = []
        for concept_id in updated_store["concept_ids"]:
            concept_vector = updated_store["concepts"][concept_id]["vector"]
            all_vectors.append(concept_vector)
            
        if all_vectors:
            updated_store["index"].add(np.array(all_vectors).astype('float32'))
    else:
        # Add new vector
        updated_store["index"].add(normalized_vector.reshape(1, -1).astype('float32'))
        updated_store["concept_ids"].append(identifier)
        updated_store["concepts"][identifier] = {
            "vector": normalized_vector,
            "metadata": metadata
        }
    
    logger.info(f"Added/updated vector with identifier: {identifier}")
    return updated_store


def get_vector_from_store(store: StoreDict, identifier: str) -> Result:
    """
    Retrieve a vector and its metadata by identifier.
    
    Args:
        store (dict): Vector store dictionary.
        identifier (str): Unique identifier for the vector.
        
    Returns:
        Result: Success with vector dictionary or error message.
    """
    if identifier not in store["concepts"]:
        return error(f"Vector with identifier '{identifier}' not found in the store")
    
    concept = store["concepts"][identifier]
    return success({
        "identifier": identifier,
        "vector": concept["vector"],
        "metadata": concept["metadata"]
    })


def create_state(session_id: str) -> StateDict:
    """
    Create a new state for a session.
    
    Args:
        session_id (str): Unique identifier for the session.
        
    Returns:
        dict: New state dictionary with standard structure.
    """
    return {
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "concepts": {},  # Maps concept identifiers to vectors and metadata
        "relationships": {},  # Maps relationship identifiers to source, target, and metadata
        "references": {},  # Maps reference identifiers to resolved entities
        "metadata": {}  # Session-specific metadata
    }


def add_concept_to_state(state: StateDict, concept: Dict[str, Any]) -> StateDict:
    """
    Add a concept to the state.
    
    Args:
        state (dict): Current state dictionary.
        concept (dict): Concept to add to the state.
        
    Returns:
        dict: Updated state dictionary with the new concept added.
    """
    # Create a copy of the state to avoid modifying the input
    updated_state = copy.deepcopy(state)
    
    # Check required fields
    if "identifier" not in concept:
        logger.error("Missing required field 'identifier' in concept")
        return updated_state
    
    identifier = concept["identifier"]
    updated_state["concepts"][identifier] = {
        "vector": concept.get("vector", None),
        "metadata": concept.get("metadata", {}),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    logger.debug(f"Added concept to state: {identifier}")
    return updated_state


def add_relation_to_state(state: StateDict, relation: Dict[str, Any]) -> StateDict:
    """
    Add a relation to the state.
    
    Args:
        state (dict): Current state dictionary.
        relation (dict): Relation to add to the state.
        
    Returns:
        dict: Updated state dictionary with the new relation added.
    """
    # Create a copy of the state to avoid modifying the input
    updated_state = copy.deepcopy(state)
    
    # Check required fields
    required_fields = ["identifier", "source", "target", "relation_type"]
    for field in required_fields:
        if field not in relation:
            logger.error(f"Missing required field '{field}' in relation")
            return updated_state
    
    # Check that source and target exist in the state
    source_id = relation["source"]
    target_id = relation["target"]
    
    if source_id not in updated_state["concepts"]:
        logger.error(f"Source concept '{source_id}' not found in state")
        return updated_state
    
    if target_id not in updated_state["concepts"]:
        logger.error(f"Target concept '{target_id}' not found in state")
        return updated_state
    
    identifier = relation["identifier"]
    updated_state["relationships"][identifier] = {
        "source": source_id,
        "target": target_id,
        "relation_type": relation["relation_type"],
        "metadata": relation.get("metadata", {}),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    logger.debug(f"Added relation to state: {identifier}")
    return updated_state


def load_or_create_vector_store(config: Dict[str, Any]) -> StoreDict:
    """
    Load an existing vector store or create a new one.
    
    Args:
        config (dict): Configuration dictionary containing processing options.
        
    Returns:
        dict: Vector store dictionary, either loaded or newly created.
    """
    persistence_options = extract_persistence_options(config)
    processing_options = extract_processing_options(config)
    
    if persistence_options["load_previous_state"] and persistence_options["previous_state_path"]:
        try:
            # In a real implementation, we would load the vector store from disk
            # For simplicity, we'll create a new one
            logger.info(f"Would load vector store from {persistence_options['previous_state_path']}")
            logger.info("Creating new vector store instead for this example")
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            logger.info("Creating new vector store instead")
    
    dimension = processing_options["vector_dimension"]
    vector_type = processing_options["vector_type"]
    index_type = "flat" if vector_type == "binary" else "hnsw"
    
    logger.info(f"Creating new vector store with dimension {dimension} and index type {index_type}")
    return create_vector_store(dimension, index_type)


def load_or_create_state(config: Dict[str, Any]) -> StateDict:
    """
    Load an existing state or create a new one.
    
    Args:
        config (dict): Configuration dictionary containing persistence options.
        
    Returns:
        dict: State dictionary, either loaded or newly created.
    """
    persistence_options = extract_persistence_options(config)
    
    if persistence_options["load_previous_state"] and persistence_options["previous_state_path"]:
        try:
            # In a real implementation, we would load the state from disk
            # For simplicity, we'll create a new one
            logger.info(f"Would load state from {persistence_options['previous_state_path']}")
            logger.info("Creating new state instead for this example")
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            logger.info("Creating new state instead")
    
    # Generate a unique session ID
    session_id = f"session_{int(time.time())}"
    
    logger.info(f"Creating new state with session ID {session_id}")
    return create_state(session_id)


def simulate_llm_rule_conversion(rule_text: str, rule_certainty: float) -> Dict[str, Any]:
    """
    Simulate LLM conversion of rule text to ACEP representation.
    In a real implementation, this would call an LLM API.
    
    Args:
        rule_text (str): The rule text to convert.
        rule_certainty (float): The certainty associated with the rule.
        
    Returns:
        dict: Simulated ACEP representation of the rule.
    """
    # This is a simplistic simulation - a real implementation would parse the rule properly
    rule_id = rule_text.lower().replace(" ", "_")[:50]
    is_conditional = "if" in rule_text.lower() and "then" in rule_text.lower()
    
    if is_conditional:
        parts = rule_text.lower().split("then")
        antecedent = parts[0].replace("if", "").strip()
        consequent = parts[1].strip().rstrip(".")
        
        # Create a vector from a hash of the rule text
        text_hash = hash(rule_text) % (2**32)
        np.random.seed(text_hash)
        vector = np.random.normal(0, 1, 10000)
        vector = vector / np.linalg.norm(vector)
        
        return {
            "identifier": f"{consequent}_if_{antecedent}"[:50],
            "vector": vector,
            "metadata": {
                "source": "rule",
                "text": rule_text,
                "certainty": rule_certainty,
                "conditional": True,
                "antecedent": antecedent,
                "consequent": consequent
            }
        }
    else:
        # Non-conditional rule
        text_hash = hash(rule_text) % (2**32)
        np.random.seed(text_hash)
        vector = np.random.normal(0, 1, 10000)
        vector = vector / np.linalg.norm(vector)
        
        return {
            "identifier": rule_id,
            "vector": vector,
            "metadata": {
                "source": "rule",
                "text": rule_text,
                "certainty": rule_certainty,
                "conditional": False
            }
        }


def simulate_llm_fact_conversion(fact_text: str, fact_certainty: float, entity_id: str) -> Dict[str, Any]:
    """
    Simulate LLM conversion of fact text to ACEP representation.
    In a real implementation, this would call an LLM API.
    
    Args:
        fact_text (str): The fact text to convert.
        fact_certainty (float): The certainty associated with the fact.
        entity_id (str): The ID of the entity the fact is about.
        
    Returns:
        dict: Simulated ACEP representation of the fact.
    """
    # This is a simplistic simulation - a real implementation would parse the fact properly
    fact_id = f"{entity_id}_{fact_text.lower().replace(' ', '_')[:30]}"
    
    # Extract a simulated assessment and metric type
    assessment = "unknown"
    metric_type = "unknown"
    
    if "high" in fact_text.lower():
        assessment = "high"
    elif "low" in fact_text.lower():
        assessment = "low"
    elif "increasing" in fact_text.lower() or "growing" in fact_text.lower():
        assessment = "positive"
    elif "decreasing" in fact_text.lower() or "declining" in fact_text.lower():
        assessment = "negative"
    
    if "p/e" in fact_text.lower() or "price to earnings" in fact_text.lower():
        metric_type = "pe_ratio"
    elif "revenue" in fact_text.lower():
        metric_type = "revenue_growth"
    elif "profit margin" in fact_text.lower():
        metric_type = "profit_margin"
    elif "debt" in fact_text.lower():
        metric_type = "debt_to_equity"
    elif "return on equity" in fact_text.lower() or "roe" in fact_text.lower():
        metric_type = "return_on_equity"
    elif "price" in fact_text.lower() and ("movement" in fact_text.lower() or "trend" in fact_text.lower()):
        metric_type = "price_movement"
    elif "analyst" in fact_text.lower():
        metric_type = "analyst_sentiment"
    
    # Create a vector from a hash of the fact text
    text_hash = hash(fact_text) % (2**32)
    np.random.seed(text_hash)
    vector = np.random.normal(0, 1, 10000)
    vector = vector / np.linalg.norm(vector)
    
    return {
        "identifier": f"{entity_id}_{metric_type}_{assessment}",
        "vector": vector,
        "metadata": {
            "source": "fact",
            "text": fact_text,
            "certainty": fact_certainty,
            "entity_id": entity_id,
            "assessment": assessment,
            "metric_type": metric_type
        }
    }


def process_rules(rules: List[Dict[str, Any]], store: StoreDict, state: StateDict, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], StoreDict, StateDict]:
    """
    Process rules and add them to the store and state.
    
    Args:
        rules (list): List of rule dictionaries from the input configuration.
        store (dict): Vector store where rule vectors will be stored.
        state (dict): State dictionary where rule representations will be added.
        config (dict): Processing configuration options.
        
    Returns:
        tuple: (processed_rules, updated_store, updated_state)
    """
    processed_rules = []
    updated_store = copy.deepcopy(store)
    updated_state = copy.deepcopy(state)
    
    for rule in rules:
        rule_text = rule.get("text", "")
        rule_certainty = rule.get("certainty", 0.9)
        
        if not rule_text:
            logger.warning("Skipping rule with empty text")
            continue
        
        # In a real implementation, this would call an LLM API
        # For simplicity, we'll use a simulated conversion
        acep_representation = simulate_llm_rule_conversion(rule_text, rule_certainty)
        
        # Add to vector store
        updated_store = add_vector_to_store(
            updated_store,
            acep_representation["identifier"],
            acep_representation["vector"],
            acep_representation["metadata"]
        )
        
        # Add to state
        updated_state = add_concept_to_state(updated_state, acep_representation)
        
        # Add to processed rules
        processed_rules.append(acep_representation)
        
        logger.info(f"Processed rule: {rule_text[:50]}...")
    
    return processed_rules, updated_store, updated_state


def process_facts(entity: Dict[str, Any], store: StoreDict, state: StateDict, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], StoreDict, StateDict]:
    """
    Process facts for an entity and add them to the store and state.
    
    Args:
        entity (dict): Entity dictionary from the input configuration.
        store (dict): Vector store where fact vectors will be stored.
        state (dict): State dictionary where fact representations will be added.
        config (dict): Processing configuration options.
        
    Returns:
        tuple: (processed_facts, updated_store, updated_state)
    """
    processed_facts = []
    updated_store = copy.deepcopy(store)
    updated_state = copy.deepcopy(state)
    
    entity_id = entity.get("id", "")
    facts = entity.get("facts", [])
    
    if not entity_id:
        logger.warning("Skipping entity with empty ID")
        return processed_facts, updated_store, updated_state
    
    if not facts:
        logger.warning(f"No facts for entity {entity_id}")
        return processed_facts, updated_store, updated_state
    
    for fact in facts:
        fact_text = fact.get("text", "")
        fact_certainty = fact.get("certainty", 0.9)
        
        if not fact_text:
            logger.warning(f"Skipping empty fact for entity {entity_id}")
            continue
        
        # In a real implementation, this would call an LLM API
        # For simplicity, we'll use a simulated conversion
        acep_representation = simulate_llm_fact_conversion(fact_text, fact_certainty, entity_id)
        
        # Add to vector store
        updated_store = add_vector_to_store(
            updated_store,
            acep_representation["identifier"],
            acep_representation["vector"],
            acep_representation["metadata"]
        )
        
        # Add to state
        updated_state = add_concept_to_state(updated_state, acep_representation)
        
        # Add to processed facts
        processed_facts.append(acep_representation)
        
        logger.info(f"Processed fact for entity {entity_id}: {fact_text[:50]}...")
    
    return processed_facts, updated_store, updated_state


def is_conditional(rule: Dict[str, Any]) -> bool:
    """Check if a rule is conditional."""
    return rule.get("metadata", {}).get("conditional", False)


def extract_antecedent(rule: Dict[str, Any]) -> str:
    """Extract the antecedent from a conditional rule."""
    return rule.get("metadata", {}).get("antecedent", "")


def matches(fact: Dict[str, Any], antecedent: str, store: StoreDict) -> bool:
    """
    Check if a fact matches a rule's antecedent.
    
    In a real implementation, this would use vector similarity or more sophisticated matching,
    leveraging the hyperdimensional computing capabilities of the system.
    For simplicity, we'll use simple string matching in this example.
    
    Args:
        fact (dict): Fact representation.
        antecedent (str): Antecedent of the rule.
        store (dict): Vector store for similarity comparisons.
        
    Returns:
        bool: True if the fact matches the antecedent, False otherwise.
    """
    # Get fact metadata
    assessment = fact.get("metadata", {}).get("assessment", "").lower()
    metric_type = fact.get("metadata", {}).get("metric_type", "").lower()
    
    # Do simple string matching
    if not antecedent or not metric_type:
        return False
    
    # Check if the metric type is mentioned in the antecedent
    metric_match = False
    if "pe ratio" in antecedent and metric_type == "pe_ratio":
        metric_match = True
    elif "revenue growth" in antecedent and metric_type == "revenue_growth":
        metric_match = True
    elif "profit margin" in antecedent and metric_type == "profit_margin":
        metric_match = True
    elif "debt" in antecedent and metric_type == "debt_to_equity":
        metric_match = True
    elif "return on equity" in antecedent and metric_type == "return_on_equity":
        metric_match = True
    elif "price movement" in antecedent and metric_type == "price_movement":
        metric_match = True
    elif "analyst" in antecedent and metric_type == "analyst_sentiment":
        metric_match = True
    
    # If metric type matches, check if assessment matches
    if metric_match:
        if "high" in antecedent and assessment == "high":
            return True
        elif "low" in antecedent and assessment == "low":
            return True
        elif "positive" in antecedent and assessment == "positive":
            return True
        elif "negative" in antecedent and assessment == "negative":
            return True
        elif "moderate" in antecedent and assessment == "moderate":
            return True
    
    # Special case for derived concepts like "undervalued"
    derived_concepts = ["undervalued", "overvalued", "strong growth", "weak growth",
                        "financially healthy", "financially weak", "positive momentum",
                        "negative momentum"]
    
    for concept in derived_concepts:
        if concept in antecedent and concept in fact.get("identifier", "").lower():
            return True
    
    return False