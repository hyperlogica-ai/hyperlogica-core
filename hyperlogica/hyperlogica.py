#!/usr/bin/env python3
"""
Hyperlogica: Main Processing Pipeline

This module implements the central processing pipeline for the Hyperlogica system,
orchestrating the flow from configuration parsing to output generation. The system
is redesigned to work with direct ACEP representations without LLM conversion.

The pipeline includes:
1. Configuration parsing
2. Vector store initialization
3. Processing of rule and fact vectors
4. Vector-chain reasoning
5. Result generation with explanations
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Import enhanced components
from .config_parser import parse_input_config, validate_config, extract_processing_options
from .config_parser import extract_persistence_options, extract_output_schema
from .vector_operations import (
    generate_vector, normalize_vector, bind_vectors, unbind_vectors,
    bundle_vectors, calculate_similarity, create_role_vectors,
    create_conditional_representation, create_fact_representation
)
from .vector_store import (
    create_store, add_vector, get_vector, filter_store_by_type,
    filter_store_by_entity, find_similar_vectors, save_store, load_store
)
from .reasoning_engine import apply_vector_chain_reasoning, generate_explanation
from .state_management import (
    create_state, add_rule_to_state, add_fact_to_state, 
    add_conclusion_to_state, add_reasoning_trace, save_state, load_state
)
from .error_handling import success, error, is_success, is_error, get_value, get_error

# Configure logging
logger = logging.getLogger(__name__)

def process_input_file(input_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process an input configuration file and generate results using hyperdimensional computing.
    
    Args:
        input_path (str): Path to the input JSON configuration file.
        options (dict, optional): Additional processing options such as:
                                 - verbose (bool): Enable verbose output
                                 - output_path (str): Where to save results
                                 - timeout (int): Processing timeout in seconds
                                 Defaults to None.
        
    Returns:
        dict: Processing results dictionary containing:
              - entities_processed (int): Number of entities processed
              - conclusions_generated (int): Number of conclusions generated
              - processing_time (float): Total processing time in seconds
              - results (list): List of result dictionaries for each entity
              
    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input file has invalid configuration.
        TimeoutError: If processing exceeds the specified timeout.
    """
    start_time = time.time()
    options = options or {}
    verbose = options.get("verbose", False)
    output_path = options.get("output_path", None)
    timeout = options.get("timeout", 3600)  # Default timeout: 1 hour
    
    try:
        # Step 1: Parse and validate configuration
        if verbose:
            print(f"Loading configuration from {input_path}...")
        
        config_result = parse_input_config(input_path)
        if is_error(config_result):
            raise ValueError(f"Failed to parse configuration: {get_error(config_result)}")
        
        config = get_value(config_result)
        validated_result = validate_config(config)
        if is_error(validated_result):
            raise ValueError(f"Failed to validate configuration: {get_error(validated_result)}")
        
        validated_config = get_value(validated_result)
        
        # Step 2: Extract configuration components
        processing_options = extract_processing_options(validated_config)
        persistence_options = extract_persistence_options(validated_config)
        output_schema = extract_output_schema(validated_config)
        
        # Get vector dimension and domain config
        vector_dim = processing_options.get("vector_dimension", 10000)
        domain_config = processing_options.get("domain_config", {})
        
        # Set up logging
        log_config = validated_config.get("logging", {})
        log_level = log_config.get("log_level", "info")
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        
        if verbose:
            print("Configuration loaded and validated successfully.")
        
        # Step 3: Initialize vector store and state
        if verbose:
            print("Initializing vector store and state...")
        
        # Create role vectors for structured binding
        roles = create_role_vectors(vector_dim)
        
        # Create or load vector store
        vector_store = None
        if persistence_options.get("load_previous_state", False):
            store_path = persistence_options.get("vector_store_path", "./output/vector_store.faiss")
            if os.path.exists(store_path):
                try:
                    vector_store = load_store(store_path)
                    if verbose:
                        print(f"Loaded vector store from {store_path}")
                except Exception as e:
                    logger.error(f"Failed to load vector store: {e}")
                    vector_store = None
        
        if vector_store is None:
            vector_store = create_store(vector_dim)
            if verbose:
                print(f"Created new vector store with dimension {vector_dim}")
        
        # Create or load state
        state = None
        if persistence_options.get("load_previous_state", False):
            state_path = persistence_options.get("state_save_path", "./output/state.pkl")
            if os.path.exists(state_path):
                try:
                    state = load_state(state_path)
                    if verbose:
                        print(f"Loaded state from {state_path}")
                except Exception as e:
                    logger.error(f"Failed to load state: {e}")
                    state = None
        
        if state is None:
            session_id = f"session_{int(time.time())}"
            state = create_state(session_id)
            if verbose:
                print(f"Created new state with session ID {session_id}")
        
        # Step 4: Process rules
        if verbose:
            print("Processing rules...")
        
        rules_data = validated_config.get("input_data", {}).get("rules", [])
        processed_rules = []
        
        for rule_idx, rule_data in enumerate(rules_data):
            # Get the ACEP representation
            acep_repr = rule_data.get("acep", {})
            certainty = rule_data.get("certainty", 0.9)
            
            # Ensure it has the right structure
            if acep_repr.get("type") != "conditional_relation":
                logger.warning(f"Rule {rule_idx} does not have type 'conditional_relation', skipping")
                continue
            
            # Extract condition and implication
            content = acep_repr.get("content", {})
            condition = content.get("condition", {})
            implication = content.get("implication", {})
            
            if not condition or not implication:
                logger.warning(f"Rule {rule_idx} is missing condition or implication, skipping")
                continue
            
            # Generate vectors for this rule
            rule_vectors = create_conditional_representation(condition, implication, roles, vector_dim)
            
            # Create the complete rule object
            rule_id = acep_repr.get("identifier", f"rule_{rule_idx}")
            rule = {
                "identifier": rule_id,
                "vector": rule_vectors["rule_vector"],
                "condition_vector": rule_vectors["condition_vector"],
                "implication_vector": rule_vectors["implication_vector"],
                "component_vectors": rule_vectors["component_vectors"],
                "acep": acep_repr,
                "attributes": {
                    "certainty": certainty,
                    "conditional": True
                }
            }
            
            # Add to the vector store
            vector_store = add_vector(vector_store, rule_id, rule_vectors["rule_vector"], acep_repr)
            
            # Add to state
            state = add_rule_to_state(state, rule)
            
            # Add to processed rules
            processed_rules.append(rule)
            
            if verbose and (rule_idx + 1) % 10 == 0:
                print(f"Processed {rule_idx + 1}/{len(rules_data)} rules")
        
        if verbose:
            print(f"Processed {len(processed_rules)} rules")
        
        # Step 5: Process entities and facts
        if verbose:
            print("Processing entities and facts...")
        
        entities_data = validated_config.get("input_data", {}).get("entities", [])
        results = []
        
        for entity_idx, entity_data in enumerate(entities_data):
            entity_id = entity_data.get("id", f"entity_{entity_idx}")
            entity_name = entity_data.get("name", entity_id)
            
            if verbose:
                print(f"Processing entity {entity_id} ({entity_idx+1}/{len(entities_data)})")
            
            # Process facts for this entity
            facts_data = entity_data.get("facts", [])
            processed_facts = []
            
            for fact_idx, fact_data in enumerate(facts_data):
                # Get the ACEP representation
                acep_repr = fact_data.get("acep", {})
                certainty = fact_data.get("certainty", 0.8)
                
                # Ensure it has the right structure
                if acep_repr.get("type") != "factual_assertion":
                    logger.warning(f"Fact {fact_idx} does not have type 'factual_assertion', skipping")
                    continue
                
                # Extract fact content
                content = acep_repr.get("content", {})
                
                # Generate vectors for this fact
                fact_vectors = create_fact_representation(content, roles, vector_dim)
                
                # Create the complete fact object
                fact_id = acep_repr.get("identifier", f"{entity_id}_fact_{fact_idx}")
                fact = {
                    "identifier": fact_id,
                    "vector": fact_vectors["fact_vector"],
                    "component_vectors": fact_vectors["component_vectors"],
                    "acep": acep_repr,
                    "attributes": {
                        "certainty": certainty,
                        "entity_id": entity_id
                    }
                }
                
                # Add to the vector store
                vector_store = add_vector(vector_store, fact_id, fact_vectors["fact_vector"], acep_repr)
                
                # Add to state
                state = add_fact_to_state(state, fact)
                
                # Add to processed facts
                processed_facts.append(fact)
            
            if verbose:
                print(f"Processed {len(processed_facts)} facts for entity {entity_id}")
            
            # Check timeout
            current_time = time.time()
            if current_time - start_time > timeout:
                raise TimeoutError(f"Processing timeout after {timeout} seconds")
            
            # Step 6: Apply vector-chain reasoning
            if verbose:
                print(f"Applying vector-chain reasoning for entity {entity_id}")
            
            max_depth = processing_options.get("max_reasoning_depth", 5)
            similarity_threshold = processing_options.get("similarity_threshold", 0.7)
            
            reasoning_result = apply_vector_chain_reasoning(
                processed_rules,
                processed_facts,
                entity_id,
                roles,
                max_depth,
                similarity_threshold
            )
            
            # Step 7: Add reasoning results to state
            for conclusion in reasoning_result.get("conclusions", []):
                state = add_conclusion_to_state(state, conclusion)
            
            # Add reasoning trace to state
            trace = {
                "entity_id": entity_id,
                "timestamp": datetime.now().isoformat(),
                "chains": reasoning_result.get("chains", []),
                "outcome": reasoning_result.get("outcome", "NEUTRAL"),
                "certainty": reasoning_result.get("certainty", 0.5)
            }
            state = add_reasoning_trace(state, trace)
            
            # Step 8: Generate explanation
            explanation = generate_explanation(reasoning_result)
            
            # Map outcome to domain-specific values
            outcome_mapping = {
                "POSITIVE": domain_config.get("positive_outcome", "BUY"),
                "NEGATIVE": domain_config.get("negative_outcome", "SELL"),
                "NEUTRAL": domain_config.get("neutral_outcome", "HOLD")
            }
            
            final_outcome = outcome_mapping.get(reasoning_result.get("outcome", "NEUTRAL"))
            
            # Step 9: Prepare entity result
            entity_result = {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "outcome": final_outcome,
                "certainty": reasoning_result.get("certainty", 0.5),
                "reasoning": {
                    "explanation": explanation,
                    "positive_signals": len(reasoning_result.get("positive_conclusions", [])),
                    "negative_signals": len(reasoning_result.get("negative_conclusions", [])),
                    "neutral_signals": len(reasoning_result.get("neutral_conclusions", [])),
                    "chains": len(reasoning_result.get("chains", [])),
                    "evidence_weights": reasoning_result.get("evidence_weights", {})
                }
            }
            
            results.append(entity_result)
            
            if verbose:
                print(f"Completed reasoning for {entity_id}: {final_outcome} with {entity_result['certainty']:.2f} certainty")
        
        # Step 10: Save state and vector store if configured
        if persistence_options.get("save_state", False):
            state_path = persistence_options.get("state_save_path", "./output/state.pkl")
            save_state(state, state_path)
            
            store_path = persistence_options.get("vector_store_path", "./output/vector_store.faiss")
            save_store(vector_store, store_path)
            
            if verbose:
                print(f"Saved state and vector store")
        
        # Step 11: Prepare final results
        processing_time = time.time() - start_time
        
        final_results = {
            "entities_processed": len(entities_data),
            "conclusions_generated": sum(r["reasoning"]["positive_signals"] + r["reasoning"]["negative_signals"] + r["reasoning"]["neutral_signals"] for r in results),
            "processing_time": processing_time,
            "results": results,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "configuration_file": input_path,
                "reasoning_approach": "vector_chain",
                "vector_dimension": vector_dim
            }
        }
        
        # Save results if output path provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Handle numpy arrays in the results
            def numpy_converter(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                raise TypeError(f"Unserializable object: {type(obj)}")
            
            # Write results to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, default=numpy_converter, indent=2)
            
            if verbose:
                print(f"Saved results to {output_path}")
        
        if verbose:
            print(f"Processing completed in {processing_time:.2f} seconds")
        
        return final_results
    
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}", exc_info=True)
        
        # Return error information
        return {
            "error": str(e),
            "entities_processed": 0,
            "conclusions_generated": 0,
            "processing_time": time.time() - start_time,
            "results": [],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperlogica Processing Pipeline")
    parser.add_argument("input_file", help="Path to the input JSON configuration file")
    parser.add_argument("--output", "-o", help="Path to save the output JSON results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=3600, help="Processing timeout in seconds")
    
    args = parser.parse_args()
    
    options = {
        "verbose": args.verbose,
        "output_path": args.output,
        "timeout": args.timeout
    }
    
    try:
        results = process_input_file(args.input_file, options)
        if args.verbose:
            print(f"Processing complete! Results for {results['entities_processed']} entities.")
            if not args.output:
                print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
