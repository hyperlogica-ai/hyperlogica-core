"""
Hyperdimensional Computing Core Module

This module integrates the pure functional implementation of vector operations,
vector store, reasoning engine, and state management to provide the core 
functionality of the Hyperlogica system.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime

# Import functional components
from .vector_operations import (
    generate_vector, normalize_vector, bind_vectors, unbind_vectors, 
    bundle_vectors, calculate_similarity, create_role_vectors, 
    create_conditional_representation, extract_conditional_parts
)
from .vector_store import (
    create_store, add_vector, get_vector, find_similar_vectors, save_store, load_store
)
from .reasoning_engine import (
    create_concept, parse_rule, is_conditional, calculate_certainty, 
    apply_modus_ponens, apply_conjunction_introduction, create_reasoning_chain,
    classify_conclusion, generate_explanation
)
from .state_management import (
    create_state, add_concept_to_state, add_relation_to_state, 
    add_conclusion_to_state, add_reasoning_trace, save_state, load_state
)

# Configure logging
logger = logging.getLogger(__name__)

def setup_hyperlogica_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up the core components of the Hyperlogica system based on configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, Any]: System components including vector store, state, and role vectors
    """
    # Extract configuration options
    processing_options = config.get("processing", {})
    vector_dimension = processing_options.get("vector_dimension", 10000)
    vector_type = processing_options.get("vector_type", "continuous")
    
    # Create vector store
    index_type = processing_options.get("index_type", "flat")
    store = create_store(dimension=vector_dimension, index_type=index_type)
    
    # Create state
    session_id = f"session_{int(time.time())}"
    state = create_state(session_id)
    
    # Create role vectors for structured binding
    role_vectors = create_role_vectors(dimension=vector_dimension)
    
    # Create signal vectors for classification
    domain_config = processing_options.get("domain_config", {})
    signal_vectors = {}
    
    # Positive outcome vector
    positive_keywords = domain_config.get("positive_outcome_keywords", ["positive", "yes", "true"])
    positive_vector = generate_vector(" ".join(positive_keywords), vector_dimension, vector_type)
    signal_vectors["positive"] = positive_vector
    
    # Negative outcome vector
    negative_keywords = domain_config.get("negative_outcome_keywords", ["negative", "no", "false"])
    negative_vector = generate_vector(" ".join(negative_keywords), vector_dimension, vector_type)
    signal_vectors["negative"] = negative_vector
    
    # Neutral outcome vector
    neutral_keywords = domain_config.get("neutral_outcome_keywords", ["neutral", "maybe", "uncertain"])
    neutral_vector = generate_vector(" ".join(neutral_keywords), vector_dimension, vector_type)
    signal_vectors["neutral"] = neutral_vector
    
    return {
        "store": store,
        "state": state,
        "role_vectors": role_vectors,
        "signal_vectors": signal_vectors,
        "config": config
    }

def process_rule(rule_data: Dict[str, Any], system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a rule and add it to the system.
    
    Args:
        rule_data (Dict[str, Any]): Rule data including text and certainty
        system (Dict[str, Any]): System components
        
    Returns:
        Dict[str, Any]: Updated system with rule added
    """
    # Extract components from system
    store = system["store"]
    state = system["state"]
    role_vectors = system["role_vectors"]
    config = system["config"]
    
    # Extract rule data
    rule_text = rule_data.get("text", "")
    rule_certainty = rule_data.get("certainty", 0.9)
    
    # Skip empty rules
    if not rule_text:
        logger.warning("Skipping empty rule")
        return system
    
    try:
        # Parse rule into antecedent and consequent
        antecedent, consequent = parse_rule(rule_text)
        
        # Create unique identifier for rule
        import hashlib
        rule_hash = hashlib.md5(rule_text.encode()).hexdigest()[:8]
        rule_id = f"rule_{rule_hash}"
        
        # Get processing options
        processing_options = config.get("processing", {})
        vector_dimension = processing_options.get("vector_dimension", 10000)
        vector_type = processing_options.get("vector_type", "continuous")
        
        # Generate vectors for antecedent and consequent
        antecedent_vector = generate_vector(antecedent, vector_dimension, vector_type)
        consequent_vector = generate_vector(consequent, vector_dimension, vector_type)
        
        # Create conditional representation
        rule_vector = create_conditional_representation(antecedent_vector, consequent_vector, role_vectors)
        
        # Create rule concept
        rule_concept = create_concept(
            identifier=rule_id,
            vector=rule_vector,
            attributes={
                "text": rule_text,
                "antecedent": antecedent,
                "consequent": consequent,
                "certainty": rule_certainty,
                "conditional": True
            },
            concept_type="rule"
        )
        
        # Add rule vector to store
        store = add_vector(store, rule_id, rule_vector, {
            "type": "rule",
            "text": rule_text,
            "antecedent": antecedent,
            "consequent": consequent,
            "certainty": rule_certainty
        })
        
        # Add rule to state
        state = add_concept_to_state(state, rule_concept)
        
        # Update system
        system["store"] = store
        system["state"] = state
        
        logger.info(f"Processed rule: {rule_id}")
        return system
        
    except Exception as e:
        logger.error(f"Failed to process rule: {str(e)}")
        return system

def process_fact(fact_data: Dict[str, Any], entity_id: str, system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a fact and add it to the system.
    
    Args:
        fact_data (Dict[str, Any]): Fact data including text and certainty
        entity_id (str): Entity identifier
        system (Dict[str, Any]): System components
        
    Returns:
        Dict[str, Any]: Updated system with fact added
    """
    # Extract components from system
    store = system["store"]
    state = system["state"]
    config = system["config"]
    
    # Extract fact data
    fact_text = fact_data.get("text", "")
    fact_certainty = fact_data.get("certainty", 0.9)
    
    # Skip empty facts
    if not fact_text:
        logger.warning("Skipping empty fact")
        return system
    
    try:
        # Create unique identifier for fact
        import hashlib
        fact_hash = hashlib.md5(fact_text.encode()).hexdigest()[:8]
        fact_id = f"{entity_id}_fact_{fact_hash}"
        
        # Get processing options
        processing_options = config.get("processing", {})
        vector_dimension = processing_options.get("vector_dimension", 10000)
        vector_type = processing_options.get("vector_type", "continuous")
        
        # Generate vector for fact
        fact_vector = generate_vector(fact_text, vector_dimension, vector_type)
        
        # Create fact concept
        fact_concept = create_concept(
            identifier=fact_id,
            vector=fact_vector,
            attributes={
                "text": fact_text,
                "certainty": fact_certainty,
                "entity_id": entity_id
            },
            concept_type="fact"
        )
        
        # Add fact vector to store
        store = add_vector(store, fact_id, fact_vector, {
            "type": "fact",
            "text": fact_text,
            "certainty": fact_certainty,
            "entity_id": entity_id
        })
        
        # Add fact to state
        state = add_concept_to_state(state, fact_concept)
        
        # Update system
        system["store"] = store
        system["state"] = state
        
        logger.info(f"Processed fact: {fact_id}")
        return system
        
    except Exception as e:
        logger.error(f"Failed to process fact: {str(e)}")
        return system

def apply_reasoning(system: Dict[str, Any], entity_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Apply reasoning to derive conclusions from rules and facts.
    
    Args:
        system (Dict[str, Any]): System components
        entity_id (str, optional): Entity ID to filter facts
        
    Returns:
        Dict[str, Any]: Updated system with reasoning results
    """
    # Extract components from system
    store = system["store"]
    state = system["state"]
    role_vectors = system["role_vectors"]
    signal_vectors = system["signal_vectors"]
    config = system["config"]
    
    # Extract processing options
    processing_options = config.get("processing", {})
    reasoning_approach = processing_options.get("reasoning_approach", "vector_weighted")
    certainty_propagation = processing_options.get("certainty_propagation", "min")
    max_reasoning_depth = processing_options.get("max_reasoning_depth", 5)
    similarity_threshold = processing_options.get("similarity_threshold", 0.7)
    
    # Extract domain configuration
    domain_config = processing_options.get("domain_config", {})
    positive_outcome = domain_config.get("positive_outcome", "POSITIVE")
    negative_outcome = domain_config.get("negative_outcome", "NEGATIVE")
    neutral_outcome = domain_config.get("neutral_outcome", "NEUTRAL")
    
    # Collect rules and facts from state
    rules = []
    facts = []
    
    for concept_id, concept in state["concepts"].items():
        if concept.get("type") == "rule":
            rules.append(concept)
        elif concept.get("type") == "fact":
            # Filter by entity_id if provided
            if entity_id is None or concept.get("attributes", {}).get("entity_id") == entity_id:
                facts.append(concept)
    
    # Skip reasoning if no facts or rules
    if not facts or not rules:
        logger.warning("No facts or rules to reason with")
        return system
    
    logger.info(f"Applying reasoning with {len(rules)} rules and {len(facts)} facts")
    
    # Apply reasoning based on approach
    if reasoning_approach == "chain":
        # Vector-based reasoning chain
        reasoning_result = create_reasoning_chain(
            initial_premises=facts,
            rules=rules,
            role_vectors=role_vectors,
            max_depth=max_reasoning_depth,
            similarity_threshold=similarity_threshold
        )
        
        # Extract conclusions
        conclusions = reasoning_result.get("conclusions", [])
        reasoning_steps = reasoning_result.get("steps", [])
        
        # Add conclusions to system
        for conclusion in conclusions:
            # Classify conclusion
            signal_type = classify_conclusion(conclusion, signal_vectors)
            conclusion["attributes"]["signal_type"] = signal_type
            
            # Add to state
            state = add_conclusion_to_state(state, conclusion)
        
        # Add reasoning trace to state
        trace = {
            "timestamp": datetime.now().isoformat(),
            "entity_id": entity_id,
            "approach": reasoning_approach,
            "steps": reasoning_steps,
            "conclusions_count": len(conclusions)
        }
        state = add_reasoning_trace(state, trace)
        
        # Create explanations dictionary for concepts
        concepts_dict = {}
        for concept_id, concept in state["concepts"].items():
            concepts_dict[concept_id] = concept
        
        # Generate explanation
        explanation = generate_explanation(conclusions, reasoning_steps, concepts_dict)
        
        # Determine outcome based on signal types
        signal_counts = {"positive": 0, "negative": 0, "neutral": 0}
        signal_certainties = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        
        for conclusion in conclusions:
            signal_type = conclusion["attributes"].get("signal_type", "neutral")
            certainty = conclusion["attributes"].get("certainty", 0.5)
            
            signal_counts[signal_type] += 1
            signal_certainties[signal_type] += certainty
        
        # Determine outcome based on total certainty
        if signal_certainties["positive"] > signal_certainties["negative"]:
            outcome = positive_outcome
            certainty = 0.5 + (signal_certainties["positive"] / 
                             (signal_certainties["positive"] + signal_certainties["negative"] + signal_certainties["neutral"])) * 0.5
        elif signal_certainties["negative"] > signal_certainties["positive"]:
            outcome = negative_outcome
            certainty = 0.5 + (signal_certainties["negative"] / 
                             (signal_certainties["positive"] + signal_certainties["negative"] + signal_certainties["neutral"])) * 0.5
        else:
            outcome = neutral_outcome
            certainty = 0.5
        
        # Create result
        result = {
            "outcome": outcome,
            "certainty": certainty,
            "signal_counts": signal_counts,
            "signal_certainties": signal_certainties,
            "conclusions": conclusions,
            "explanation": explanation,
            "entity_id": entity_id
        }
        
    else:  # weighted or other approaches
        # Initialize evidence tracking
        positive_evidence = 0.0
        negative_evidence = 0.0
        neutral_evidence = 0.0
        derived_conclusions = []
        
        # Try to derive conclusions by applying rules to facts
        for rule in rules:
            for fact in facts:
                try:
                    # Apply modus ponens
                    conclusion = apply_modus_ponens(
                        rule=rule,
                        fact=fact,
                        role_vectors=role_vectors,
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Classify conclusion
                    signal_type = classify_conclusion(conclusion, signal_vectors)
                    conclusion["attributes"]["signal_type"] = signal_type
                    
                    # Add to derived conclusions
                    derived_conclusions.append(conclusion)
                    
                    # Add to evidence
                    certainty = conclusion["attributes"].get("certainty", 0.5)
                    if signal_type == "positive":
                        positive_evidence += certainty
                    elif signal_type == "negative":
                        negative_evidence += certainty
                    else:
                        neutral_evidence += certainty
                    
                    # Add to state
                    state = add_conclusion_to_state(state, conclusion)
                    
                except ValueError:
                    # Rule doesn't apply, skip
                    pass
        
        # Calculate total evidence
        total_evidence = positive_evidence + negative_evidence + neutral_evidence
        
        # Determine outcome based on evidence
        if total_evidence == 0:
            outcome = neutral_outcome
            certainty = 0.5
        elif positive_evidence > negative_evidence:
            outcome = positive_outcome
            certainty = 0.5 + (positive_evidence / total_evidence) * 0.5
        elif negative_evidence > positive_evidence:
            outcome = negative_outcome
            certainty = 0.5 + (negative_evidence / total_evidence) * 0.5
        else:
            outcome = neutral_outcome
            certainty = 0.5
        
        # Create reasoning steps for explanation
        reasoning_steps = []
        for i, conclusion in enumerate(derived_conclusions):
            step = {
                "step_id": i + 1,
                "pattern": "modus_ponens",
                "premises": conclusion["attributes"]["derived_from"],
                "conclusion": conclusion["identifier"],
                "certainty": conclusion["attributes"]["certainty"]
            }
            reasoning_steps.append(step)
        
        # Create explanations dictionary for concepts
        concepts_dict = {}
        for concept_id, concept in state["concepts"].items():
            concepts_dict[concept_id] = concept
        
        # Generate explanation
        explanation = generate_explanation(derived_conclusions, reasoning_steps, concepts_dict)
        
        # Add reasoning trace to state
        trace = {
            "timestamp": datetime.now().isoformat(),
            "entity_id": entity_id,
            "approach": reasoning_approach,
            "steps": reasoning_steps,
            "conclusions_count": len(derived_conclusions),
            "evidence": {
                "positive": positive_evidence,
                "negative": negative_evidence,
                "neutral": neutral_evidence
            },
            "outcome": outcome,
            "certainty": certainty
        }
        state = add_reasoning_trace(state, trace)
        
        # Create result
        result = {
            "outcome": outcome,
            "certainty": certainty,
            "evidence_weights": {
                "positive": positive_evidence,
                "negative": negative_evidence,
                "neutral": neutral_evidence
            },
            "conclusions": derived_conclusions,
            "explanation": explanation,
            "entity_id": entity_id
        }
    
    # Update system
    system["state"] = state
    system["result"] = result
    
    logger.info(f"Applied reasoning, outcome: {outcome}, certainty: {certainty:.4f}")
    return system

def process_entities(entities: List[Dict[str, Any]], system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process multiple entities and their facts, applying reasoning to each.
    
    Args:
        entities (List[Dict[str, Any]]): List of entities with facts
        system (Dict[str, Any]): System components
        
    Returns:
        Dict[str, Any]: Updated system with all entities processed
    """
    # Create results list
    entity_results = []
    
    # Process each entity
    for entity in entities:
        entity_id = entity.get("id", "")
        entity_name = entity.get("name", entity_id)
        entity_facts = entity.get("facts", [])
        
        # Skip entities without ID
        if not entity_id:
            logger.warning("Skipping entity without ID")
            continue
        
        logger.info(f"Processing entity: {entity_id}")
        
        # Process facts for this entity
        for fact in entity_facts:
            system = process_fact(fact, entity_id, system)
        
        # Apply reasoning for this entity
        system = apply_reasoning(system, entity_id)
        
        # Get reasoning result
        result = system.get("result", {})
        
        # Add entity information
        result["entity_id"] = entity_id
        result["entity_name"] = entity_name
        
        # Add to results
        entity_results.append(result)
    
    # Add to system
    system["entity_results"] = entity_results
    
    return system

def save_system_state(system: Dict[str, Any], state_path: str, store_path: str) -> Dict[str, Any]:
    """
    Save the system state and vector store to disk.
    
    Args:
        system (Dict[str, Any]): System components
        state_path (str): Path to save state
        store_path (str): Path to save vector store
        
    Returns:
        Dict[str, Any]: System with save status
    """
    # Create a copy of the system
    system_copy = system.copy()
    
    # Save state
    state = system["state"]
    state_saved = save_state(state, state_path)
    system_copy["state_saved"] = state_saved
    
    # Save vector store
    store = system["store"]
    store_saved = save_store(store, store_path)
    system_copy["store_saved"] = store_saved
    
    return system_copy

def process_input_file(input_path: Optional[str] = None, 
                      options: Optional[Dict[str, Any]] = None,
                      config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process an input configuration file using hyperdimensional computing.
    
    Args:
        input_path (str, optional): Path to input configuration file
        options (Dict[str, Any], optional): Additional options
        config_dict (Dict[str, Any], optional): Configuration dictionary (if not loading from file)
        
    Returns:
        Dict[str, Any]: Results dictionary
    """
    options = options or {}
    verbose = options.get("verbose", False)
    output_path = options.get("output_path", None)
    
    start_time = time.time()
    
    try:
        # Load configuration
        if config_dict:
            config = config_dict
        elif input_path:
            with open(input_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError("Either input_path or config_dict must be provided")
        
        if verbose:
            print("Setting up Hyperlogica system...")
        
        # Setup system
        system = setup_hyperlogica_system(config)
        
        # Process rules
        if verbose:
            print("Processing rules...")
        
        rules_data = config.get("input_data", {}).get("rules", [])
        for rule_data in rules_data:
            system = process_rule(rule_data, system)
        
        # Process entities
        if verbose:
            print("Processing entities...")
        
        entities_data = config.get("input_data", {}).get("entities", [])
        system = process_entities(entities_data, system)
        
        # Save state if configured
        persistence_options = config.get("persistence", {})
        if persistence_options.get("save_state", False):
            if verbose:
                print("Saving system state...")
                
            state_path = persistence_options.get("state_save_path", "./output/state.pkl")
            store_path = persistence_options.get("vector_store_path", "./output/vector_store.faiss")
            system = save_system_state(system, state_path, store_path)
        
        # Prepare results
        entity_results = system.get("entity_results", [])
        processing_time = time.time() - start_time
        
        results = {
            "entities_processed": len(entity_results),
            "conclusions_generated": sum(len(result.get("conclusions", [])) for result in entity_results),
            "processing_time": processing_time,
            "results": entity_results,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "reasoning_approach": config.get("processing", {}).get("reasoning_approach", "vector_weighted"),
                "vector_dimension": config.get("processing", {}).get("vector_dimension", 10000)
            }
        }
        
        # Save results if output path provided
        if output_path:
            if verbose:
                print(f"Saving results to {output_path}...")
                
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Convert numpy arrays to lists
            from .state_management import convert_numpy_to_lists
            serializable_results = convert_numpy_to_lists(results)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        if verbose:
            print(f"Processing completed in {processing_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise