#!/usr/bin/env python3
"""
Hyperlogica: Main Processing Pipeline

This module implements the central processing pipeline for the Hyperlogica system,
orchestrating the flow from configuration parsing to output generation. It integrates
vector operations, LLM interfacing, reasoning, and state management.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Import components
from config_parser import parse_input_config, validate_config, extract_processing_options
from config_parser import extract_persistence_options, extract_output_schema
from vector_operations import generate_vector, normalize_vector
from vector_store import create_store, add_vector, get_vector, save_store, load_store
from llm_interface import convert_english_to_acep, convert_acep_to_english
from reasoning_engine import apply_modus_ponens, calculate_certainty, recalibrate_certainty
from state_management import create_state, add_concept_to_state, add_relation_to_state, save_state, load_state
from logging_utils import initialize_logger, log_reasoning_step, log_llm_interaction
from reasoning_approaches import apply_reasoning_approach
from error_handling import success, error, is_success, is_error, get_value, get_error


def process_input_file(input_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process an input configuration file and generate results.
    
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
        validated_config = validate_config(config)
        
        # Extract different parts of the configuration
        processing_options = extract_processing_options(validated_config)
        persistence_options = extract_persistence_options(validated_config)
        output_schema = extract_output_schema(validated_config)
        
        # Set up logging
        log_config = validated_config.get("logging", {})
        logger = initialize_logger(
            log_path=log_config.get("log_path", "./logs"),
            log_level=log_config.get("log_level", "info")
        )
        logger.info(f"Configuration loaded from {input_path}")
        
        # Step 2: Initialize vector store and state
        vector_store = load_or_create_vector_store(persistence_options)
        state = load_or_create_state(persistence_options)
        
        # Step 3: Process rules
        rules_data = validated_config.get("input_data", {}).get("rules", [])
        llm_options = validated_config.get("llm", {})
        
        if verbose:
            print(f"Processing {len(rules_data)} rules...")
            
        logger.info(f"Processing {len(rules_data)} rules...")
        processed_rules = process_rules(rules_data, llm_options, vector_store, state, processing_options)
        logger.info(f"Processed {len(processed_rules)} rules")
        
        # Step 4: Process entities and facts
        entities_data = validated_config.get("input_data", {}).get("entities", [])
        entity_count = len(entities_data)
        
        if verbose:
            print(f"Processing {entity_count} entities...")
            
        logger.info(f"Processing {entity_count} entities...")
        
        results = []
        conclusions_count = 0
        
        for entity_index, entity in enumerate(entities_data):
            entity_id = entity.get("id", f"entity_{entity_index}")
            entity_name = entity.get("name", entity_id)
            entity_facts = entity.get("facts", [])
            
            if verbose:
                print(f"Processing entity {entity_id} ({entity_index+1}/{entity_count})")
                
            logger.info(f"Processing entity {entity_id} ({entity_index+1}/{entity_count}) with {len(entity_facts)} facts")
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Processing timeout after {timeout} seconds")
            
            # Process entity facts
            processed_facts = process_facts(entity_facts, llm_options, vector_store, state, processing_options, entity_id)
            logger.info(f"Processed {len(processed_facts)} facts for entity {entity_id}")
            
            # Apply reasoning to derive conclusions
            reasoning_approach = processing_options.get("reasoning_approach", "majority")
            
            if verbose:
                print(f"Applying {reasoning_approach} reasoning approach...")
                
            logger.info(f"Applying {reasoning_approach} reasoning approach...")
            
            reasoning_result = apply_reasoning(
                processed_rules, 
                processed_facts, 
                reasoning_approach, 
                vector_store, 
                state, 
                processing_options
            )
            
            # Generate output for this entity
            entity_conclusions = reasoning_result.get("conclusions", [])
            conclusions_count += len(entity_conclusions)
            
            if verbose:
                print(f"Generated {len(entity_conclusions)} conclusions for {entity_id}")
                
            logger.info(f"Generated {len(entity_conclusions)} conclusions for {entity_id}")
            
            entity_output = generate_entity_output(
                entity_id, 
                entity_name, 
                reasoning_result, 
                output_schema, 
                llm_options
            )
            
            results.append(entity_output)
        
        # Step 5: Save state if configured
        if persistence_options.get("save_state", False):
            state_save_path = persistence_options.get("state_save_path", "./state.pkl")
            logger.info(f"Saving state to {state_save_path}")
            save_state(state, state_save_path)
        
        # Step 6: Compile overall results
        processing_time = time.time() - start_time
        
        final_results = {
            "entities_processed": entity_count,
            "conclusions_generated": conclusions_count,
            "processing_time": processing_time,
            "results": results,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "configuration_file": input_path,
                "reasoning_approach": reasoning_approach,
                "vector_dimension": processing_options.get("vector_dimension", 10000)
            }
        }
        
        # Save results if output path provided
        if output_path:
            if verbose:
                print(f"Saving results to {output_path}...")
                
            logger.info(f"Saving results to {output_path}")
            save_results(final_results, output_path)
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        if verbose:
            print(f"Processing completed in {processing_time:.2f} seconds")
        
        return final_results
    
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}", exc_info=True)
        raise


def load_or_create_vector_store(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load an existing vector store or create a new one.
    
    Args:
        config (dict): Configuration dictionary containing persistence options 
                      and vector store settings.
        
    Returns:
        dict: Vector store dictionary, either loaded from disk or newly created.
        
    Raises:
        ValueError: If configuration is invalid.
        FileNotFoundError: If attempting to load a non-existent store.
    """
    load_previous_store = config.get("load_previous_store", False)
    previous_store_path = config.get("previous_store_path", "")
    vector_dimension = config.get("vector_dimension", 10000)
    index_type = config.get("index_type", "flat")
    
    if load_previous_store and os.path.exists(previous_store_path):
        logging.info(f"Loading existing vector store from {previous_store_path}")
        try:
            return load_store(previous_store_path)
        except Exception as e:
            logging.warning(f"Failed to load vector store: {str(e)}. Creating new store.")
    
    logging.info(f"Creating new vector store with dimension={vector_dimension}, index_type={index_type}")
    return create_store(dimension=vector_dimension, index_type=index_type)


def load_or_create_state(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load an existing state or create a new one.
    
    Args:
        config (dict): Configuration dictionary containing persistence options
                      and state management settings.
        
    Returns:
        dict: State dictionary, either loaded from disk or newly created.
        
    Raises:
        ValueError: If configuration is invalid.
        FileNotFoundError: If attempting to load a non-existent state.
    """
    load_previous_state = config.get("load_previous_state", False)
    previous_state_path = config.get("previous_state_path", "")
    session_id = config.get("session_id", f"session_{int(time.time())}")
    
    if load_previous_state and os.path.exists(previous_state_path):
        logging.info(f"Loading existing state from {previous_state_path}")
        try:
            return load_state(previous_state_path)
        except Exception as e:
            logging.warning(f"Failed to load state: {str(e)}. Creating new state.")
    
    logging.info(f"Creating new state with session_id={session_id}")
    return create_state(session_id)


def process_rules(rules: List[Dict[str, Any]], 
                 llm_options: Dict[str, Any], 
                 store: Dict[str, Any], 
                 state: Dict[str, Any], 
                 config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process rules and add them to the store and state.
    
    Args:
        rules (list): List of rule dictionaries from the input configuration.
        llm_interface (dict): LLM interface configuration for converting rules to ACEP.
        store (dict): Vector store where rule vectors will be stored.
        state (dict): State dictionary where rule representations will be added.
        config (dict): Processing configuration options.
        
    Returns:
        list: List of processed rule representations with vector IDs and metadata.
        
    Raises:
        ValueError: If rule processing fails.
        OpenAIError: If LLM API calls fail.
    """
    processed_rules = []
    
    for rule_index, rule in enumerate(rules):
        rule_text = rule.get("text", "")
        rule_certainty = rule.get("certainty", 0.9)  # Default high certainty for rules
        
        if not rule_text:
            logging.warning(f"Skipping empty rule at index {rule_index}")
            continue
        
        logging.info(f"Processing rule [{rule_index+1}/{len(rules)}]: {rule_text[:50]}...")
        
        try:
            # Convert rule to ACEP representation
            rule_context = {
                "domain": config.get("domain", "general"),
                "certainty": rule_certainty
            }
            
            acep_representation = convert_english_to_acep(rule_text, rule_context, llm_options)
            
            # Ensure certainty from original rule is preserved
            acep_representation["attributes"]["certainty"] = rule_certainty
            
            # Generate or retrieve vector for the rule
            vector_dimension = config.get("vector_dimension", 10000)
            
            if "vector" not in acep_representation:
                vector = generate_vector(acep_representation["identifier"], vector_dimension)
                acep_representation["vector"] = normalize_vector(vector)
            
            # Add to vector store
            add_vector(
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
            add_concept_to_state(state, acep_representation)
            
            # Save to processed rules
            processed_rules.append(acep_representation)
            
            logging.info(f"Rule processed successfully: {acep_representation['identifier']}")
            
        except Exception as e:
            logging.error(f"Failed to process rule: {str(e)}")
            # Continue processing other rules
    
    return processed_rules


def process_facts(facts: List[Dict[str, Any]], 
                 llm_options: Dict[str, Any], 
                 store: Dict[str, Any], 
                 state: Dict[str, Any], 
                 config: Dict[str, Any],
                 entity_id: str) -> List[Dict[str, Any]]:
    """
    Process facts and add them to the store and state.
    
    Args:
        facts (list): List of fact dictionaries from the input configuration.
        llm_interface (dict): LLM interface configuration for converting facts to ACEP.
        store (dict): Vector store where fact vectors will be stored.
        state (dict): State dictionary where fact representations will be added.
        config (dict): Processing configuration options.
        entity_id (str): Identifier of the entity to which the facts belong.
        
    Returns:
        list: List of processed fact representations with vector IDs and metadata.
        
    Raises:
        ValueError: If fact processing fails.
        OpenAIError: If LLM API calls fail.
    """
    processed_facts = []
    
    for fact_index, fact in enumerate(facts):
        fact_text = fact.get("text", "")
        fact_certainty = fact.get("certainty", 0.9)  # Default high certainty for facts
        
        if not fact_text:
            logging.warning(f"Skipping empty fact at index {fact_index}")
            continue
        
        logging.info(f"Processing fact [{fact_index+1}/{len(facts)}] for entity {entity_id}: {fact_text[:50]}...")
        
        try:
            # Convert fact to ACEP representation
            fact_context = {
                "domain": config.get("domain", "general"),
                "entity_id": entity_id,
                "certainty": fact_certainty
            }
            
            acep_representation = convert_english_to_acep(fact_text, fact_context, llm_options)
            
            # Ensure certainty from original fact is preserved
            acep_representation["attributes"]["certainty"] = fact_certainty
            
            # Ensure entity_id is associated with the fact
            acep_representation["attributes"]["entity_id"] = entity_id
            
            # Generate or retrieve vector for the fact
            vector_dimension = config.get("vector_dimension", 10000)
            
            if "vector" not in acep_representation:
                vector = generate_vector(acep_representation["identifier"], vector_dimension)
                acep_representation["vector"] = normalize_vector(vector)
            
            # Add to vector store
            add_vector(
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
            
            # Add to state
            add_concept_to_state(state, acep_representation)
            
            # Save to processed facts
            processed_facts.append(acep_representation)
            
            logging.info(f"Fact processed successfully: {acep_representation['identifier']}")
            
        except Exception as e:
            logging.error(f"Failed to process fact: {str(e)}")
            # Continue processing other facts
    
    return processed_facts


def apply_reasoning(rules: List[Dict[str, Any]], 
                   facts: List[Dict[str, Any]], 
                   approach: str, 
                   store: Dict[str, Any], 
                   state: Dict[str, Any], 
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply reasoning to derive conclusions.
    
    Args:
        rules (list): List of processed rule representations.
        facts (list): List of processed fact representations.
        approach (str): Reasoning approach to use (e.g., "majority", "weighted", "bayesian").
        store (dict): Vector store containing rule and fact vectors.
        state (dict): State dictionary for tracking reasoning context.
        config (dict): Configuration dictionary containing reasoning settings.
        
    Returns:
        dict: Dictionary containing derived conclusions, including:
              - outcome (str): Final outcome or recommendation
              - certainty (float): Overall certainty in the outcome
              - conclusions (list): List of intermediate conclusions
              - reasoning_trace (dict): Detailed record of reasoning steps
              
    Raises:
        ValueError: If an unsupported reasoning approach is specified or if reasoning fails.
    """
    logging.info(f"Applying reasoning approach: {approach}")
    
    # Filter facts to only include those relevant to the current entity if needed
    entity_id = facts[0]["attributes"]["entity_id"] if facts else None
    
    # Apply the selected reasoning approach
    try:
        result = apply_reasoning_approach(approach, rules, facts, store, state, config)
        
        # Apply recalibration if enabled
        if config.get("recalibration_enabled", False):
            recalibration_method = config.get("recalibration_method", "linear")
            context = {
                "domain": config.get("domain", "general"),
                "entity_id": entity_id
            }
            
            original_certainty = result.get("certainty", 0.5)
            recalibrated_certainty = recalibrate_certainty(original_certainty, context, recalibration_method)
            
            result["certainty"] = recalibrated_certainty
            result["original_certainty"] = original_certainty
            
            logging.info(f"Recalibrated certainty: {original_certainty:.4f} → {recalibrated_certainty:.4f}")
        
        return result
        
    except Exception as e:
        error_msg = f"Reasoning failed with approach '{approach}': {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)


def generate_entity_output(entity_id: str, 
                          entity_name: str, 
                          reasoning_result: Dict[str, Any], 
                          output_schema: Dict[str, Any],
                          llm_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate output for an entity according to the specified schema.
    
    Args:
        entity_id (str): Identifier of the entity.
        entity_name (str): Name of the entity.
        reasoning_result (dict): Results from the reasoning process.
        output_schema (dict): Output schema specification from configuration.
        llm_options (dict): LLM options for generating natural language explanations.
        
    Returns:
        dict: Formatted output dictionary for the entity.
    """
    output = {
        "entity_id": entity_id,
        "entity_name": entity_name,
        "outcome": reasoning_result.get("outcome", "UNKNOWN"),
        "certainty": reasoning_result.get("certainty", 0.5)
    }
    
    # Add fields based on output schema
    fields = output_schema.get("fields", [])
    
    for field in fields:
        field_name = field.get("name", "")
        field_type = field.get("type", "string")
        
        # Skip already added fields
        if field_name in output:
            continue
        
        # Handle special field types
        if field_name == "reasoning":
            output[field_name] = generate_reasoning_output(reasoning_result, output_schema, llm_options)
        elif field_name in reasoning_result:
            output[field_name] = reasoning_result[field_name]
    
    # Include reasoning trace if requested
    if output_schema.get("include_reasoning_trace", False):
        output["reasoning_trace"] = reasoning_result.get("reasoning_trace", {})
    
    return output


def generate_reasoning_output(reasoning_result: Dict[str, Any], 
                             output_schema: Dict[str, Any],
                             llm_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate reasoning output according to the schema.
    
    Args:
        reasoning_result (dict): Results from the reasoning process.
        output_schema (dict): Output schema specification.
        llm_options (dict): LLM options for generating natural language explanations.
        
    Returns:
        dict: Formatted reasoning output.
    """
    reasoning_output = {}
    
    # Add signal counts if available
    if "signal_counts" in reasoning_result:
        reasoning_output["positive_signals"] = reasoning_result["signal_counts"].get("positive", 0)
        reasoning_output["negative_signals"] = reasoning_result["signal_counts"].get("negative", 0)
        reasoning_output["neutral_signals"] = reasoning_result["signal_counts"].get("neutral", 0)
    
    # Add evidence weights if available
    if "evidence_weights" in reasoning_result:
        reasoning_output["evidence_weights"] = reasoning_result["evidence_weights"]
    
    # Add posteriors if available
    if "posteriors" in reasoning_result:
        reasoning_output["posteriors"] = reasoning_result["posteriors"]
    
    # Generate key factors
    conclusions = reasoning_result.get("conclusions", [])
    key_factors = []
    
    for conclusion in conclusions:
        factor = {
            "factor": conclusion["identifier"],
            "certainty": conclusion.get("attributes", {}).get("certainty", 0.5)
        }
        
        # Determine impact if possible
        if "signal_type" in conclusion.get("attributes", {}):
            factor["impact"] = conclusion["attributes"]["signal_type"]
        
        key_factors.append(factor)
    
    reasoning_output["key_factors"] = key_factors
    
    # Generate natural language explanation if LLM is available
    reasoning_trace = reasoning_result.get("reasoning_trace", None)
    
    if reasoning_trace and output_schema.get("include_explanation", True):
        try:
            explanation_context = {
                "domain": output_schema.get("domain", "general"),
                "entity_id": reasoning_result.get("entity_id", ""),
                "recommendation": reasoning_result.get("outcome", ""),
                "certainty": reasoning_result.get("certainty", 0.5)
            }
            
            from llm_interface import generate_explanation
            explanation = generate_explanation(reasoning_trace, explanation_context, llm_options)
            reasoning_output["explanation"] = explanation
            
        except Exception as e:
            logging.warning(f"Failed to generate explanation: {str(e)}")
    
    return reasoning_output


def save_results(output: Dict[str, Any], output_path: str) -> bool:
    """
    Save results to the specified output path.
    
    Args:
        output (dict): Output dictionary to save.
        output_path (str): File path where results should be saved.
        
    Returns:
        bool: True if results were successfully saved, False otherwise.
        
    Raises:
        IOError: If the output directory doesn't exist or isn't writable.
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write the results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        return False


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
