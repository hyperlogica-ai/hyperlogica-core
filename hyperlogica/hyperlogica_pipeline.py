"""
Hyperlogica Pipeline Module with Ontology Support

Implements the main processing pipeline with ontology-based standardization.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from functools import reduce

from .config_parser import parse_input_config, validate_config, extract_processing_options, extract_persistence_options, extract_output_schema
from .vector_operations import generate_vector, normalize_vector, bind_vectors, unbind_vectors, bundle_vectors, calculate_similarity
from .vector_store import create_store, add_vector, get_vector, save_store, load_store
from .ontology_mapper import create_ontology_mapper, standardize_rule_text, standardize_fact_text
from .llm_interface import process_rules_with_ontology, process_facts_with_ontology
from .reasoning_approaches import apply_reasoning_approach
from .state_management import create_state, add_concept_to_state, add_relation_to_state, add_conclusion_to_state, save_state, load_state
from .error_handling import success, error, is_success, is_error, get_value, get_error

# Configure logging
logger = logging.getLogger(__name__)

def process_input_file(input_path=None, options=None, config_dict=None):
    """Process an input configuration file with ontology support."""
    start_time = time.time()
    options = options or {}
    verbose = options.get("verbose", False)
    output_path = options.get("output_path", None)
    timeout = options.get("timeout", 3600)  # Default timeout: 1 hour
    
    try:
        # Step 1: Load and validate configuration
        if verbose:
            print("Loading and validating configuration...")
        
        if config_dict:
            config = config_dict
        elif input_path:
            config_result = parse_input_config(input_path)
            if is_error(config_result):
                raise ValueError(f"Failed to parse configuration: {get_error(config_result)}")
            config = get_value(config_result)
        else:
            raise ValueError("Either input_path or config_dict must be provided")
        
        validated_result = validate_config(config)
        if is_error(validated_result):
            raise ValueError(f"Invalid configuration: {get_error(validated_result)}")
        validated_config = get_value(validated_result)
        
        # Extract configuration components
        processing_options = extract_processing_options(validated_config)
        persistence_options = extract_persistence_options(validated_config)
        output_schema = extract_output_schema(validated_config)
        llm_options = validated_config.get("llm", {})
        domain_config = processing_options.get("domain_config", {})
        
        # Step 2: Initialize vector store and state
        if verbose:
            print("Initializing vector store and state...")
        
        vector_dim = processing_options.get("vector_dimension", 10000)
        vector_store = load_or_create_vector_store(persistence_options, processing_options)
        state = load_or_create_state(persistence_options)
        
        # Step 3: Process rules with ontology support
        if verbose:
            print("Processing rules with ontology support...")
        
        rules_data = validated_config.get("input_data", {}).get("rules", [])
        processed_rules = process_rules_with_ontology(
            rules_data, 
            llm_options, 
            {**processing_options, "domain_config": domain_config}
        )
        
        for rule in processed_rules:
            # Add rule to vector store
            vector_store = add_vector(
                vector_store,
                rule["identifier"],
                rule["vector"],
                {
                    "type": "rule",
                    "acep": rule
                }
            )
            
            # Add rule to state
            state = add_concept_to_state(state, rule)
        
        if verbose:
            print(f"Processed {len(processed_rules)} rules")
        
        # Step 4: Process entities and their facts
        if verbose:
            print("Processing entities and facts...")
        
        entities_data = validated_config.get("input_data", {}).get("entities", [])
        results = []
        
        for entity in entities_data:
            entity_id = entity.get("id", "")
            entity_name = entity.get("name", entity_id)
            entity_facts = entity.get("facts", [])
            
            if verbose:
                print(f"Processing entity: {entity_id}")
            
            # Process facts with ontology support
            processed_facts = process_facts_with_ontology(
                entity_facts,
                entity_id,
                llm_options,
                {**processing_options, "domain_config": domain_config}
            )
            
            for fact in processed_facts:
                # Add fact to vector store
                vector_store = add_vector(
                    vector_store,
                    fact["identifier"],
                    fact["vector"],
                    {
                        "type": "fact",
                        "entity_id": entity_id,
                        "acep": fact
                    }
                )
                
                # Add fact to state
                state = add_concept_to_state(state, fact)
            
            # Apply reasoning to entity facts
            reasoning_approach = processing_options.get("reasoning_approach", "weighted")
            
            reasoning_result = apply_reasoning_approach(
                reasoning_approach,
                processed_rules,
                processed_facts,
                vector_store,
                state,
                processing_options
            )
            
            # Add conclusions to state
            for conclusion in reasoning_result.get("conclusions", []):
                state = add_conclusion_to_state(state, conclusion)
            
            # Prepare entity result
            entity_result = {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "outcome": reasoning_result.get("outcome", "UNKNOWN"),
                "certainty": reasoning_result.get("certainty", 0.5),
                "reasoning": {
                    "conclusions_count": len(reasoning_result.get("conclusions", [])),
                    "reasoning_approach": reasoning_approach
                }
            }
            
            # Add additional reasoning details if available
            if "evidence_weights" in reasoning_result:
                entity_result["reasoning"]["evidence_weights"] = reasoning_result["evidence_weights"]
                entity_result["reasoning"]["positive_signals"] = len(reasoning_result.get("positive_signals", []))
                entity_result["reasoning"]["negative_signals"] = len(reasoning_result.get("negative_signals", []))
            
            if "posteriors" in reasoning_result:
                entity_result["reasoning"]["posteriors"] = reasoning_result["posteriors"]
            
            if "explanation" in reasoning_result:
                entity_result["reasoning"]["explanation"] = reasoning_result["explanation"]
            
            # Add entity result to results list
            results.append(entity_result)
            
            if verbose:
                print(f"Processed entity {entity_id}: {entity_result['outcome']} ({entity_result['certainty']:.2f})")
        
        # Save state if configured
        if persistence_options.get("save_state", False):
            state_path = persistence_options.get("state_save_path", "./output/state.pkl")
            save_state(state, state_path)
            
            if verbose:
                print(f"Saved state to {state_path}")
        
        # Save vector store if configured
        if persistence_options.get("save_store", False):
            store_path = persistence_options.get("vector_store_path", "./output/vector_store.faiss")
            save_store(vector_store, store_path)
            
            if verbose:
                print(f"Saved vector store to {store_path}")
        
        # Prepare final results
        processing_time = time.time() - start_time
        conclusions_count = sum(r.get("reasoning", {}).get("conclusions_count", 0) for r in results)
        
        final_results = {
            "entities_processed": len(entities_data),
            "conclusions_generated": conclusions_count,
            "processing_time": processing_time,
            "results": results,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "configuration_file": input_path if input_path else "in-memory-config",
                "reasoning_approach": reasoning_approach,
                "vector_dimension": vector_dim,
                "ontology_enabled": True,
                "domain": processing_options.get("domain", "general")
            }
        }
        
        # Save results if output path provided
        if output_path:
            save_results(final_results, output_path)
            
            if verbose:
                print(f"Saved results to {output_path}")
        
        if verbose:
            print(f"Processing completed in {processing_time:.2f} seconds")
        
        return final_results
    
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}", exc_info=True)
        
        if verbose:
            print(f"Error: {str(e)}")
        
        # Return error result
        return {
            "error": str(e),
            "processing_time": time.time() - start_time,
            "success": False
        }

def load_or_create_vector_store(persistence_config, processing_config):
    """Load an existing vector store or create a new one."""
    load_previous_store = persistence_config.get("load_previous_store", False)
    previous_store_path = persistence_config.get("vector_store_path", "")
    vector_dimension = processing_config.get("vector_dimension", 10000)
    index_type = processing_config.get("index_type", "flat")
    
    if load_previous_store and previous_store_path and os.path.exists(previous_store_path):
        logger.info(f"Loading vector store from {previous_store_path}")
        try:
            return load_store(previous_store_path)
        except Exception as e:
            logger.warning(f"Failed to load vector store: {str(e)}. Creating new store.")
    
    logger.info(f"Creating new vector store with dimension={vector_dimension}, index_type={index_type}")
    return create_store(dimension=vector_dimension, index_type=index_type)

def load_or_create_state(persistence_config):
    """Load an existing state or create a new one."""
    load_previous_state = persistence_config.get("load_previous_state", False)
    previous_state_path = persistence_config.get("state_save_path", "")
    
    if load_previous_state and previous_state_path and os.path.exists(previous_state_path):
        logger.info(f"Loading state from {previous_state_path}")
        try:
            return load_state(previous_state_path)
        except Exception as e:
            logger.warning(f"Failed to load state: {str(e)}. Creating new state.")
    
    session_id = f"session_{int(time.time())}"
    logger.info(f"Creating new state with session_id={session_id}")
    return create_state(session_id)

def save_results(results, output_path):
    """Save results to file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert NumPy arrays in results to lists for JSON serialization
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
