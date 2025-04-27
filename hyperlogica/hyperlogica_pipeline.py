"""
Hyperlogica Processing Pipeline

This module contains the core processing pipeline functions for the Hyperlogica system,
implemented using functional programming principles. This focuses on the main pipeline
that orchestrates the flow of data through the system, from input parsing through
reasoning to output generation using hyperdimensional vector operations.

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

# Import enhanced components
from .vector_operations import (
    generate_vector, normalize_vector, bind_vectors, unbind_vectors,
    bundle_vectors, calculate_similarity, cleanse_vector
)
from .reasoning_engine import (
    apply_modus_ponens, calculate_certainty, recalibrate_certainty
)
from .reasoning_approaches import apply_reasoning_approach
from .llm_interface import convert_english_to_acep, convert_acep_to_english, generate_explanation
from .state_management import (
    create_state, add_concept_to_state, add_relation_to_state, 
    add_conclusion_to_state, save_state, load_state
)

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
    required_sections = ["processing", "input_data"]
    for section in required_sections:
        if section not in validated:
            return error(f"Missing required configuration section: {section}")
    
    # Set defaults for processing options
    if "processing" in validated:
        processing = validated["processing"]
        processing.setdefault("vector_dimension", 10000)
        processing.setdefault("vector_type", "binary")
        processing.setdefault("reasoning_approach", "vector_weighted")  # Updated default
        processing.setdefault("certainty_propagation", "min")
        processing.setdefault("recalibration_enabled", True)
        processing.setdefault("max_reasoning_depth", 10)
        processing.setdefault("domain_config", {})
        
        # Set domain-specific defaults if needed
        domain_config = processing["domain_config"]
        domain_config.setdefault("positive_outcome_keywords", ["positive", "increase", "growth"])
        domain_config.setdefault("negative_outcome_keywords", ["negative", "decrease", "decline"])
        domain_config.setdefault("neutral_outcome_keywords", ["neutral", "stable", "unchanged"])
        domain_config.setdefault("similarity_threshold", 0.7)
    
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
    logging_opts.setdefault("include_vector_operations", True)  # Updated default
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
    
    # Set defaults for output schema
    if "output_schema" not in validated:
        validated["output_schema"] = {}
    output_schema = validated["output_schema"]
    output_schema.setdefault("format", "json")
    output_schema.setdefault("include_reasoning_trace", True)  # Updated default
    output_schema.setdefault("include_vector_details", False)
    output_schema.setdefault("include_explanation", True)
    
    if "fields" not in output_schema:
        output_schema["fields"] = [
            {"name": "entity_id", "type": "string"},
            {"name": "entity_name", "type": "string"},
            {"name": "outcome", "type": "string"},
            {"name": "certainty", "type": "float"},
            {"name": "reasoning", "type": "object"}
        ]
    
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
        "reasoning_approach": processing.get("reasoning_approach", "vector_weighted"),  # Updated default
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
        "include_vector_operations": logging_opts.get("include_vector_operations", True),  # Updated default
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
        "include_reasoning_trace": output_schema.get("include_reasoning_trace", True),  # Updated default
        "include_explanation": output_schema.get("include_explanation", True),
        "include_vector_details": output_schema.get("include_vector_details", False)
    }


def create_vector_store(dimension: int, index_type: str = "flat") -> StoreDict:
    """
    Create a new vector store with FAISS for high-dimensional vector operations.
    
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


def process_entity(entity: EntityDict, rules: List[RuleDict], 
                  store: StoreDict, state: StateDict, config: ConfigDict) -> ResultDict:
    """
    Process a single entity and its facts to generate a result using vector-based reasoning.
    
    Args:
        entity (dict): Entity to process with its facts
        rules (list): List of rule representations
        store (dict): Vector store for similarity search
        state (dict): State dictionary for context
        config (dict): Configuration options
        
    Returns:
        dict: Processing result for this entity
    """
    entity_id = entity.get("id", "")
    entity_name = entity.get("name", entity_id)
    entity_facts = entity.get("facts", [])
    
    logger.info(f"Processing entity {entity_id} with {len(entity_facts)} facts")
    
    # Extract processing and LLM options
    processing_options = extract_processing_options(config)
    llm_options = extract_llm_options(config)
    
    # Process facts using ACEP and vector representations
    facts_context = {
        "entity_id": entity_id,
        "domain": processing_options.get("domain", "general"),
        "vector_dimension": processing_options.get("vector_dimension", 10000)
    }
    
    processed_facts = []
    for fact in entity_facts:
        fact_text = fact.get("text", "")
        fact_certainty = fact.get("certainty", 0.9)
        
        # Skip empty facts
        if not fact_text:
            continue
            
        try:
            # Convert to ACEP representation
            acep_representation = convert_english_to_acep(
                fact_text, 
                {**facts_context, "certainty": fact_certainty}, 
                llm_options
            )
            
            # Add entity_id if not already present
            if "entity_id" not in acep_representation.get("attributes", {}):
                acep_representation["attributes"]["entity_id"] = entity_id
                
            # Add to store and state
            store = add_vector_to_store(
                store,
                acep_representation["identifier"],
                acep_representation["vector"],
                {
                    "text": fact_text,
                    "type": "fact",
                    "entity_id": entity_id,
                    "certainty": fact_certainty,
                    "acep": acep_representation
                }
            )
            
            state = add_concept_to_state(state, acep_representation)
            
            # Add to processed facts
            processed_facts.append(acep_representation)
            
        except Exception as e:
            logger.error(f"Error processing fact: {str(e)}")
    
    # Apply reasoning using the selected approach
    reasoning_approach = processing_options.get("reasoning_approach", "vector_weighted")
    
    try:
        reasoning_result = apply_reasoning_approach(
            reasoning_approach,
            rules,
            processed_facts,
            store,
            state,
            processing_options
        )
        
        # Add conclusions to state
        for conclusion in reasoning_result.get("conclusions", []):
            try:
                state = add_conclusion_to_state(state, conclusion)
            except Exception as e:
                logger.error(f"Error adding conclusion to state: {str(e)}")
        
        # Generate entity output
        entity_output = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "outcome": reasoning_result.get("outcome", "UNKNOWN"),
            "certainty": reasoning_result.get("certainty", 0.5),
            "reasoning": {
                "conclusions_count": len(reasoning_result.get("conclusions", [])),
                "reasoning_approach": reasoning_approach
            }
        }
        
        # Add additional reasoning information based on approach
        if "evidence_weights" in reasoning_result:
            entity_output["reasoning"]["evidence_weights"] = reasoning_result["evidence_weights"]
            
        if "posteriors" in reasoning_result:
            entity_output["reasoning"]["posteriors"] = reasoning_result["posteriors"]
            
        if "chain_weights" in reasoning_result:
            entity_output["reasoning"]["chain_weights"] = reasoning_result["chain_weights"]
            
        # Generate explanation if requested
        if config.get("output_schema", {}).get("include_explanation", True):
            explanation_context = {
                "domain": processing_options.get("domain", "general"),
                "entity_id": entity_id,
                "recommendation": reasoning_result.get("outcome", ""),
                "certainty": reasoning_result.get("certainty", 0.5)
            }
            
            try:
                explanation = generate_explanation(reasoning_result, explanation_context, llm_options)
                entity_output["reasoning"]["explanation"] = explanation
            except Exception as e:
                logger.error(f"Error generating explanation: {str(e)}")
        
        return entity_output
        
    except Exception as e:
        logger.error(f"Error in reasoning: {str(e)}")
        return {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "outcome": "ERROR",
            "certainty": 0.0,
            "error": str(e)
        }


def process_rules(rules_data: List[Dict[str, Any]], 
                 llm_options: Dict[str, Any], 
                 store: StoreDict, 
                 state: StateDict, 
                 config: ConfigDict) -> Tuple[List[RuleDict], StoreDict, StateDict]:
    """
    Process rules and add them to the store and state with vector representations.
    
    Args:
        rules_data (list): Raw rule data from the configuration
        llm_options (dict): LLM interface options
        store (dict): Vector store for similarity search
        state (dict): State dictionary for context
        config (dict): Configuration options
        
    Returns:
        tuple: (processed_rules, updated_store, updated_state)
    """
    processed_rules = []
    
    # Extract processing options
    vector_dimension = config.get("vector_dimension", 10000)
    
    for rule in rules_data:
        rule_text = rule.get("text", "")
        rule_certainty = rule.get("certainty", 0.9)
        
        # Skip empty rules
        if not rule_text:
            continue
            
        try:
            # Convert to ACEP representation
            rule_context = {
                "domain": config.get("domain", "general"),
                "certainty": rule_certainty,
                "vector_dimension": vector_dimension
            }
            
            acep_representation = convert_english_to_acep(rule_text, rule_context, llm_options)
            
            # Ensure certainty is preserved
            acep_representation["attributes"]["certainty"] = rule_certainty
            
            # Add to store
            store = add_vector_to_store(
                store,
                acep_representation["identifier"],
                acep_representation["vector"],
                {
                    "text": rule_text,
                    "type": "rule",
                    "certainty": rule_certainty,
                    "acep": acep_representation
                }
            )
            
            # Add to state
            state = add_concept_to_state(state, acep_representation)
            
            # Add to processed rules
            processed_rules.append(acep_representation)
            
            logger.info(f"Processed rule: {acep_representation['identifier']}")
            
        except Exception as e:
            logger.error(f"Error processing rule: {str(e)}")
    
    return processed_rules, store, state


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
    normalized_vector = normalize_vector(vector)
    
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


def process_entities_parallel(entities: List[EntityDict], rules: List[RuleDict],
                            store: StoreDict, state: StateDict, config: ConfigDict) -> List[ResultDict]:
    """
    Process entities in parallel using a thread pool.
    
    Args:
        entities (list): List of entities to process
        rules (list): List of rule representations
        store (dict): Vector store for similarity search
        state (dict): State dictionary for context
        config (dict): Configuration options
        
    Returns:
        list: List of processing results for entities
    """
    # Extract max workers from config or use default
    max_workers = config.get("max_workers", min(32, os.cpu_count() + 4))
    
    # Create a copy of store and state for each worker to avoid concurrency issues
    entity_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function for each entity
        futures = []
        for entity in entities:
            # Create a deep copy of store and state for each entity
            entity_store = copy.deepcopy(store)
            entity_state = copy.deepcopy(state)
            
            # Process entity with its own store and state
            future = executor.submit(
                process_entity,
                entity=entity,
                rules=rules,
                store=entity_store,
                state=entity_state,
                config=config
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                entity_results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")
                # Add error result
                entity_results.append({
                    "entity_id": "unknown",
                    "entity_name": "unknown",
                    "outcome": "ERROR",
                    "certainty": 0.0,
                    "error": str(e)
                })
    
    return entity_results


def process_pipeline(config: ConfigDict, verbose: bool = False) -> ResultDict:
    """
    Process the entire Hyperlogica pipeline from input to output.
    
    Args:
        config (dict): Configuration dictionary
        verbose (bool): Whether to print verbose progress information
        
    Returns:
        dict: Results dictionary
    """
    start_time = time.time()
    
    # Extract options
    processing_options = extract_processing_options(config)
    persistence_options = extract_persistence_options(config)
    llm_options = extract_llm_options(config)
    logging_options = extract_logging_options(config)
    output_schema = extract_output_schema(config)
    
    # Initialize vector store and state
    vector_dim = processing_options.get("vector_dimension", 10000)
    vector_store = create_vector_store(dimension=vector_dim)
    state = create_state(f"session_{int(time.time())}")
    
    # Process rules
    rules_data = config.get("input_data", {}).get("rules", [])
    
    if verbose:
        print(f"Processing {len(rules_data)} rules...")
        
    processed_rules, vector_store, state = process_rules(
        rules_data, llm_options, vector_store, state, processing_options
    )
    
    # Process entities
    entities_data = config.get("input_data", {}).get("entities", [])
    entity_count = len(entities_data)
    
    if verbose:
        print(f"Processing {entity_count} entities...")
    
    # Choose between parallel or sequential processing
    if processing_options.get("parallel_processing", True) and entity_count > 1:
        entity_results = process_entities_parallel(
            entities_data, processed_rules, vector_store, state, processing_options
        )
    else:
        entity_results = []
        for entity in entities_data:
            result = process_entity(
                entity, processed_rules, vector_store, state, processing_options
            )
            entity_results.append(result)
    
    # Save state if requested
    if persistence_options.get("save_state", False):
        state_path = persistence_options.get("state_save_path", "./state.pkl")
        try:
            save_state(state, state_path)
            if verbose:
                print(f"State saved to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Compile results
    conclusions_count = sum(
        len(result.get("reasoning", {}).get("conclusions", []))
        for result in entity_results
        if "reasoning" in result
    )
    
    final_results = {
        "entities_processed": entity_count,
        "conclusions_generated": conclusions_count,
        "processing_time": processing_time,
        "results": entity_results,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "reasoning_approach": processing_options.get("reasoning_approach", "vector_weighted"),
            "vector_dimension": vector_dim
        }
    }
    
    if verbose:
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Processed {entity_count} entities with {conclusions_count} conclusions")
    
    return final_results


def run_pipeline_from_file(file_path: str, output_path: Optional[str] = None, verbose: bool = False) -> ResultDict:
    """
    Run the Hyperlogica pipeline from a configuration file.
    
    Args:
        file_path (str): Path to the configuration file
        output_path (str, optional): Path to save the results. Defaults to None.
        verbose (bool, optional): Whether to print verbose progress information. Defaults to False.
        
    Returns:
        dict: Results dictionary
        
    Raises:
        ValueError: If the configuration file is invalid
    """
    # Parse and validate configuration
    config_result = parse_input_config(file_path)
    
    if is_error(config_result):
        raise ValueError(f"Error loading configuration: {get_error(config_result)}")
    
    config = get_value(config_result)
    
    # Validate configuration
    validated_result = validate_config(config)
    
    if is_error(validated_result):
        raise ValueError(f"Invalid configuration: {get_error(validated_result)}")
    
    validated_config = get_value(validated_result)
    
    # Run the pipeline
    results = process_pipeline(validated_config, verbose)
    
    # Save results if output path is provided
    if output_path:
        try:
            # Handle numpy arrays for JSON serialization
            def numpy_converter(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                raise TypeError(f"Unserializable object: {type(obj)}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, default=numpy_converter, indent=2)
                
            if verbose:
                print(f"Results saved to {output_path}")
                
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    return results
