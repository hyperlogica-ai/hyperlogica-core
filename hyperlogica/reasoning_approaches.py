"""
Reasoning Approaches Module
===========================

This module implements domain-agnostic reasoning approaches for the Hyperlogica system
using hyperdimensional vector operations. Each approach applies different strategies
for deriving conclusions from rules and facts using vector-based operations.

The module follows functional programming principles with pure functions and explicit state passing.
"""

import re
import numpy as np
from typing import Dict, List, Any, Callable, Union, Tuple, Optional
import logging
from functools import reduce

# Import vector operations
from .vector_operations import (
    calculate_similarity, bind_vectors, bundle_vectors, normalize_vector,
    unbind_vectors, generate_vector
)

from .reasoning_engine import (
    is_conditional, calculate_certainty,
    recalibrate_certainty
)

# Configure logger
logger = logging.getLogger(__name__)

# Registry for reasoning approaches
reasoning_approaches: Dict[str, Callable] = {}

def register_reasoning_approach(name: str):
    """
    Decorator to register a reasoning approach function.
    
    Args:
        name (str): Name to register the approach under
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable):
        reasoning_approaches[name] = func
        logger.debug(f"Registered reasoning approach: {name}")
        return func
    return decorator


def apply_reasoning_approach(approach_name: str, rules: List[Dict], facts: List[Dict], 
                            store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Apply the specified reasoning approach.
    
    Args:
        approach_name (str): Name of the reasoning approach to use
        rules (list): List of processed rule representations
        facts (list): List of processed fact representations
        store (dict): Vector store containing rule and fact vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary containing reasoning settings
        
    Returns:
        dict: Results of applying the reasoning approach, including outcome, 
              certainty, and supporting information
        
    Raises:
        ValueError: If an unknown reasoning approach is specified
    """
    if approach_name not in reasoning_approaches:
        registered_approaches = ", ".join(reasoning_approaches.keys())
        raise ValueError(f"Unknown reasoning approach: {approach_name}. Registered approaches: {registered_approaches}")
    
    logger.info(f"Applying reasoning approach: {approach_name}")
    return reasoning_approaches[approach_name](rules, facts, store, state, config)


def facts_match_rule_vector(rule_vector: np.ndarray, fact_vectors: List[np.ndarray], 
                          threshold: float = 0.7) -> bool:
    """
    Check if any fact vector matches a rule vector based on similarity.
    
    Args:
        rule_vector (np.ndarray): Vector representation of the rule
        fact_vectors (List[np.ndarray]): List of fact vector representations
        threshold (float, optional): Similarity threshold for considering a match.
                                    Defaults to 0.7.
        
    Returns:
        bool: True if any fact vector matches the rule vector, False otherwise
    """
    for fact_vector in fact_vectors:
        similarity = calculate_similarity(rule_vector, fact_vector)
        if similarity >= threshold:
            return True
    return False


def classify_signal_type(conclusion: Dict[str, Any], domain_config: Dict[str, Any]) -> str:
    """
    Classify a conclusion as positive, negative, or neutral based on text and vector analysis.
    
    Args:
        conclusion (dict): Conclusion representation to classify
        domain_config (dict): Domain-specific configuration containing signal keywords
        
    Returns:
        str: Signal type ("positive", "negative", "neutral")
    """
    # Check if signal type is already defined in the conclusion
    if "signal_type" in conclusion.get("attributes", {}):
        return conclusion["attributes"]["signal_type"]
    
    # Extract keywords for each signal type
    positive_keywords = domain_config.get("positive_outcome_keywords", [])
    negative_keywords = domain_config.get("negative_outcome_keywords", [])
    neutral_keywords = domain_config.get("neutral_outcome_keywords", [])
    
    # Get conclusion text to analyze
    conclusion_text = ""
    if "text" in conclusion.get("attributes", {}):
        conclusion_text = conclusion["attributes"]["text"].lower()
    elif "identifier" in conclusion:
        conclusion_text = conclusion["identifier"].lower()
    
    # Check for keyword matches
    positive_matches = sum(1 for word in positive_keywords if word.lower() in conclusion_text)
    negative_matches = sum(1 for word in negative_keywords if word.lower() in conclusion_text)
    neutral_matches = sum(1 for word in neutral_keywords if word.lower() in conclusion_text)
    
    # Determine signal type based on keyword matches
    if positive_matches > negative_matches and positive_matches > neutral_matches:
        return "positive"
    elif negative_matches > positive_matches and negative_matches > neutral_matches:
        return "negative"
    else:
        return "neutral"


@register_reasoning_approach("vector_weighted")
def vector_weighted_approach(rules: List[Dict], facts: List[Dict], 
                           store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Vector-based weighted reasoning approach that uses hyperdimensional computing
    to match rules to facts and derive conclusions with appropriate certainty.
    
    Args:
        rules (list): List of rule dictionaries with vector representations
        facts (list): List of fact dictionaries with vector representations
        store (dict): Vector store for retrieving related vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary with similarity threshold and domain config
        
    Returns:
        dict: Results dictionary containing:
              - outcome (str): Final recommendation based on weighted evidence
              - certainty (float): Confidence in the outcome (0.5-1.0)
              - conclusions (list): List of derived conclusions with vectors
              - evidence_weights (dict): Positive, negative, and neutral evidence weights
    """
    logger.info("Applying vector-weighted reasoning approach")
    
    # Extract domain configuration
    domain_config = config.get("domain_config", {})
    positive_outcome = domain_config.get("positive_outcome", "POSITIVE")
    negative_outcome = domain_config.get("negative_outcome", "NEGATIVE")
    neutral_outcome = domain_config.get("neutral_outcome", "NEUTRAL")
    
    # Get similarity threshold from config with improved default
    similarity_threshold = config.get("similarity_threshold", 0.65)
    
    # Get entity ID from facts if available
    entity_id = None
    if facts and "attributes" in facts[0]:
        entity_id = facts[0]["attributes"].get("entity_id", "")
    
    # Initialize evidence tracking
    positive_evidence = 0.0
    negative_evidence = 0.0
    neutral_evidence = 0.0
    conclusions = []
    vector_dimension = config.get("vector_dimension", 10000)
    
    # Process conditional rules with vector operations
    for rule_idx, rule in enumerate(rules):
        rule_id = rule.get("identifier", f"rule_{rule_idx}")
        logger.debug(f"Processing rule: {rule_id}")
        
        if not is_conditional(rule) or "vector" not in rule:
            logger.warning(f"Skipping rule {rule_id}: not conditional or missing vector")
            continue
        
        # For rules with component vectors, use unbinding for more accurate matching
        component_vectors = rule.get("component_vectors", {})
        if "antecedent" in component_vectors:
            antecedent_vector = component_vectors["antecedent"]
            consequent_vector = component_vectors["consequent"]
            has_components = True
        else:
            has_components = False
        
        # Process each fact for this rule
        for fact_idx, fact in enumerate(facts):
            fact_id = fact.get("identifier", f"fact_{fact_idx}")
            
            # Skip facts without vectors
            if "vector" not in fact:
                logger.warning(f"Skipping fact {fact_id}: missing vector")
                continue
            
            fact_vector = fact["vector"]
            
            # Extract antecedent text for logging
            try:
                antecedent_text = extract_antecedent(rule)
                logger.debug(f"Rule antecedent: {antecedent_text}")
            except ValueError:
                antecedent_text = "unknown"
            
            # Check if fact matches the rule's antecedent using vector similarity
            if has_components:
                # Use direct similarity with antecedent vector for more precise matching
                similarity = calculate_similarity(fact_vector, antecedent_vector, method="cosine")
            else:
                # Fallback to older method
                match_result, similarity = matches(fact, antecedent_text, store)
            
            # Only proceed if similarity exceeds threshold
            if similarity >= similarity_threshold:
                logger.info(f"Match found! Rule {rule_id} matches fact {fact_id} with similarity {similarity:.4f}")
                
                try:
                    # Extract consequent for the conclusion
                    consequent_text = extract_consequent(rule)
                    logger.debug(f"Rule consequent: {consequent_text}")
                    
                    # Create a vector for the conclusion using vector operations
                    if has_components:
                        # Use the precomputed consequent vector for better accuracy
                        conclusion_vector = consequent_vector
                    else:
                        # Fallback to unbinding (less accurate if vectors were randomly generated)
                        conclusion_vector = unbind_vectors(rule["vector"], fact_vector)
                        conclusion_vector = cleanse_vector(conclusion_vector)
                    
                    # Create a unique identifier for the conclusion
                    conclusion_id = f"conclusion_{entity_id}_{len(conclusions)+1}"
                    
                    # Calculate certainty with adjusted formula to account for similarity distribution
                    rule_certainty = rule.get("attributes", {}).get("certainty", 0.9)
                    fact_certainty = fact.get("attributes", {}).get("certainty", 0.9)
                    
                    # Adjust similarity to emphasize genuine matches
                    # Transform similarity from [threshold, 1.0] to [0.0, 1.0]
                    adjusted_similarity = (similarity - similarity_threshold) / (1.0 - similarity_threshold)
                    adjusted_similarity = max(0.0, min(1.0, adjusted_similarity))
                    
                    certainty = min(rule_certainty, fact_certainty) * adjusted_similarity
                    
                    # Create the conclusion
                    conclusion = {
                        "identifier": conclusion_id,
                        "type": "concept",
                        "vector": conclusion_vector,
                        "attributes": {
                            "derived_from": [rule_id, fact_id],
                            "derivation_method": "vector_modus_ponens",
                            "rule_text": rule.get("attributes", {}).get("rule_text", ""),
                            "fact_text": fact.get("attributes", {}).get("fact_text", ""),
                            "entity_id": entity_id,
                            "certainty": certainty,
                            "text": consequent_text
                        }
                    }
                    
                    # Classify the conclusion as positive, negative, or neutral
                    signal_type = classify_signal_type(conclusion, domain_config)
                    conclusion["attributes"]["signal_type"] = signal_type
                    
                    # Add to conclusions list
                    conclusions.append(conclusion)
                    logger.info(f"Generated conclusion: {conclusion_id} with certainty {certainty:.4f}, signal type: {signal_type}")
                    
                    # Update evidence weights
                    evidence_weight = certainty
                    if signal_type == "positive":
                        positive_evidence += evidence_weight
                        logger.debug(f"Added positive evidence: {evidence_weight:.4f}")
                    elif signal_type == "negative":
                        negative_evidence += evidence_weight
                        logger.debug(f"Added negative evidence: {evidence_weight:.4f}")
                    else:
                        neutral_evidence += evidence_weight
                        logger.debug(f"Added neutral evidence: {evidence_weight:.4f}")
                
                except Exception as e:
                    logger.error(f"Error processing match: {str(e)}")
    
    # Calculate final outcome with improved differential
    total_evidence = positive_evidence + negative_evidence + neutral_evidence
    logger.info(f"Evidence weights - Positive: {positive_evidence:.4f}, Negative: {negative_evidence:.4f}, Neutral: {neutral_evidence:.4f}")
    
    if total_evidence == 0:
        logger.info("No evidence found, defaulting to neutral outcome")
        return {
            "outcome": neutral_outcome,
            "certainty": 0.5,
            "conclusions": conclusions,
            "entity_id": entity_id,
            "evidence_weights": {
                "positive": positive_evidence,
                "negative": negative_evidence,
                "neutral": neutral_evidence
            }
        }
    
    # Calculate outcome with more distinctive certainty
    positive_ratio = positive_evidence / total_evidence
    negative_ratio = negative_evidence / total_evidence
    
    # Require stronger evidence for positive/negative outcomes
    if positive_ratio > negative_ratio and positive_ratio > 0.4:
        outcome = positive_outcome
        # Non-linear certainty that emphasizes strong evidence
        certainty = 0.5 + (positive_ratio ** 2) * 0.5
        logger.info(f"Positive evidence dominates: {positive_evidence:.4f} > {negative_evidence:.4f}, certainty: {certainty:.4f}")
    elif negative_ratio > positive_ratio and negative_ratio > 0.4:
        outcome = negative_outcome
        certainty = 0.5 + (negative_ratio ** 2) * 0.5
        logger.info(f"Negative evidence dominates: {negative_evidence:.4f} > {positive_evidence:.4f}, certainty: {certainty:.4f}")
    else:
        outcome = neutral_outcome
        # Calculate uncertainty based on how close positive and negative evidence are
        evidence_diff = abs(positive_evidence - negative_evidence) / total_evidence
        certainty = 0.5 + evidence_diff * 0.3
        logger.info("Evidence is mixed, neutral outcome")
    
    return {
        "outcome": outcome,
        "certainty": certainty,
        "conclusions": conclusions,
        "entity_id": entity_id,
        "evidence_weights": {
            "positive": positive_evidence,
            "negative": negative_evidence,
            "neutral": neutral_evidence
        }
    }


@register_reasoning_approach("vector_bayesian")
def vector_bayesian_approach(rules: List[Dict], facts: List[Dict], 
                            store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Vector-based Bayesian reasoning approach that uses hyperdimensional computing
    and updates posterior probabilities of outcomes based on evidence.
    
    Args:
        rules (list): List of rule dictionaries with vector representations
        facts (list): List of fact dictionaries with vector representations 
        store (dict): Vector store for retrieving related vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary with priors and domain config
        
    Returns:
        dict: Results dictionary containing:
              - outcome (str): Final recommendation based on Bayesian reasoning
              - certainty (float): Confidence in the outcome (0.0-1.0)
              - conclusions (list): List of derived conclusions with vectors
              - posteriors (dict): Posterior probabilities for each possible outcome
              - update_steps (list): Step-by-step record of Bayesian updates
    """
    logger.info("Applying vector-based Bayesian reasoning approach")
    
    # Extract domain configuration
    domain_config = config.get("domain_config", {})
    positive_outcome = domain_config.get("positive_outcome", "POSITIVE")
    negative_outcome = domain_config.get("negative_outcome", "NEGATIVE")
    neutral_outcome = domain_config.get("neutral_outcome", "NEUTRAL")
    
    # Extract Bayesian priors (or use defaults)
    prior_positive = domain_config.get("prior_positive", 1/3)
    prior_negative = domain_config.get("prior_negative", 1/3)
    prior_neutral = domain_config.get("prior_neutral", 1/3)
    
    # Normalize priors to ensure they sum to 1.0
    prior_sum = prior_positive + prior_negative + prior_neutral
    if prior_sum > 0:
        prior_positive /= prior_sum
        prior_negative /= prior_sum
        prior_neutral /= prior_sum
    else:
        prior_positive = prior_negative = prior_neutral = 1/3
    
    logger.info(f"Initial priors - Positive: {prior_positive:.4f}, Negative: {prior_negative:.4f}, Neutral: {prior_neutral:.4f}")
    
    # Get similarity threshold from config
    similarity_threshold = config.get("similarity_threshold", 0.7)
    
    # Get entity ID from facts if available
    entity_id = None
    if facts and "attributes" in facts[0]:
        entity_id = facts[0]["attributes"].get("entity_id", "")
    
    # Initialize tracking variables
    conclusions = []
    update_steps = []
    vector_dimension = config.get("vector_dimension", 10000)
    
    # Initialize posteriors with priors
    posterior_positive = prior_positive
    posterior_negative = prior_negative 
    posterior_neutral = prior_neutral
    
    # Process conditional rules with vector operations
    for rule_idx, rule in enumerate(rules):
        rule_id = rule.get("identifier", f"rule_{rule_idx}")
        logger.debug(f"Processing rule: {rule_id}")
        
        if not is_conditional(rule) or "vector" not in rule:
            logger.warning(f"Skipping rule {rule_id}: not conditional or missing vector")
            continue
        
        # Get rule vector
        rule_vector = rule["vector"]
        
        # Process each fact for this rule
        for fact_idx, fact in enumerate(facts):
            fact_id = fact.get("identifier", f"fact_{fact_idx}")
            
            # Skip facts without vectors
            if "vector" not in fact:
                logger.warning(f"Skipping fact {fact_id}: missing vector")
                continue
            
            fact_vector = fact["vector"]
            
            # Extract antecedent text for logging
            try:
                antecedent_text = extract_antecedent(rule)
                logger.debug(f"Rule antecedent: {antecedent_text}")
            except ValueError:
                antecedent_text = "unknown"
            
            # Check if fact matches the rule's antecedent using vector similarity
            match_result, similarity = matches(fact, antecedent_text, store)
            
            if match_result:
                logger.info(f"Match found! Rule {rule_id} matches fact {fact_id} with similarity {similarity:.4f}")
                
                try:
                    # Extract consequent for the conclusion
                    consequent_text = extract_consequent(rule)
                    logger.debug(f"Rule consequent: {consequent_text}")
                    
                    # Create a vector for the conclusion using vector operations
                    conclusion_vector = unbind_vectors(rule_vector, fact_vector)
                    conclusion_vector = cleanse_vector(conclusion_vector)
                    
                    # Create a unique identifier for the conclusion
                    conclusion_id = f"conclusion_{entity_id}_{len(conclusions)+1}"
                    
                    # Calculate certainty for the conclusion
                    rule_certainty = rule.get("attributes", {}).get("certainty", 0.9)
                    fact_certainty = fact.get("attributes", {}).get("certainty", 0.9)
                    match_certainty = similarity
                    certainty = min(rule_certainty, fact_certainty) * match_certainty
                    
                    # Create the conclusion with its vector representation
                    conclusion = {
                        "identifier": conclusion_id,
                        "type": "concept",
                        "vector": conclusion_vector,
                        "attributes": {
                            "derived_from": [rule_id, fact_id],
                            "derivation_method": "vector_modus_ponens",
                            "rule_text": rule.get("attributes", {}).get("rule_text", ""),
                            "fact_text": fact.get("attributes", {}).get("fact_text", ""),
                            "entity_id": entity_id,
                            "certainty": certainty,
                            "text": consequent_text
                        }
                    }
                    
                    # Classify the conclusion as positive, negative, or neutral
                    signal_type = classify_signal_type(conclusion, domain_config)
                    conclusion["attributes"]["signal_type"] = signal_type
                    
                    # Add to our conclusions list
                    conclusions.append(conclusion)
                    logger.info(f"Generated conclusion: {conclusion_id} with certainty {certainty:.4f}, signal type: {signal_type}")
                    
                    # Capture current state before update
                    pre_update = {
                        "step": len(update_steps) + 1,
                        "rule_id": rule_id,
                        "fact_id": fact_id,
                        "conclusion_id": conclusion_id,
                        "signal_type": signal_type,
                        "evidence_certainty": certainty,
                        "prior_positive": posterior_positive,
                        "prior_negative": posterior_negative,
                        "prior_neutral": posterior_neutral
                    }
                    
                    # Prepare likelihoods for Bayesian update
                    # P(evidence|hypothesis) for each hypothesis
                    if signal_type == "positive":
                        likelihood_positive = certainty
                        likelihood_negative = 1.0 - certainty
                        likelihood_neutral = 0.5
                    elif signal_type == "negative":
                        likelihood_positive = 1.0 - certainty
                        likelihood_negative = certainty
                        likelihood_neutral = 0.5
                    else:  # neutral
                        likelihood_positive = 0.5
                        likelihood_negative = 0.5
                        likelihood_neutral = certainty
                    
                    # Calculate denominator for Bayes' rule
                    # P(evidence) = Sum_h P(evidence|h)P(h)
                    denominator = (
                        posterior_positive * likelihood_positive +
                        posterior_negative * likelihood_negative +
                        posterior_neutral * likelihood_neutral
                    )
                    
                    # Update posteriors with Bayes' rule if denominator is valid
                    if denominator > 0:
                        new_posterior_positive = (posterior_positive * likelihood_positive) / denominator
                        new_posterior_negative = (posterior_negative * likelihood_negative) / denominator
                        new_posterior_neutral = (posterior_neutral * likelihood_neutral) / denominator
                        
                        posterior_positive = new_posterior_positive
                        posterior_negative = new_posterior_negative
                        posterior_neutral = new_posterior_neutral
                        
                        logger.debug(f"Updated posteriors - Positive: {posterior_positive:.4f}, "
                                   f"Negative: {posterior_negative:.4f}, Neutral: {posterior_neutral:.4f}")
                    else:
                        logger.warning(f"Skipping Bayesian update due to zero denominator")
                    
                    # Record the posteriors after update
                    post_update = {
                        "likelihood_positive": likelihood_positive,
                        "likelihood_negative": likelihood_negative,
                        "likelihood_neutral": likelihood_neutral,
                        "posterior_positive": posterior_positive,
                        "posterior_negative": posterior_negative,
                        "posterior_neutral": posterior_neutral
                    }
                    
                    # Add to update steps
                    update_steps.append({**pre_update, **post_update})
                    
                except Exception as e:
                    logger.error(f"Error processing match: {str(e)}")
    
    # Determine final outcome based on posterior probabilities
    logger.info(f"Final posteriors - Positive: {posterior_positive:.4f}, "
               f"Negative: {posterior_negative:.4f}, Neutral: {posterior_neutral:.4f}")
    
    if posterior_positive > posterior_negative and posterior_positive > posterior_neutral:
        outcome = positive_outcome
        certainty = posterior_positive
        logger.info(f"Highest posterior for positive outcome: {outcome}, certainty: {certainty:.4f}")
    elif posterior_negative > posterior_positive and posterior_negative > posterior_neutral:
        outcome = negative_outcome
        certainty = posterior_negative
        logger.info(f"Highest posterior for negative outcome: {outcome}, certainty: {certainty:.4f}")
    else:
        outcome = neutral_outcome
        certainty = posterior_neutral
        logger.info(f"Highest posterior for neutral outcome: {outcome}, certainty: {certainty:.4f}")
    
    return {
        "outcome": outcome,
        "certainty": certainty,
        "conclusions": conclusions,
        "entity_id": entity_id,
        "posteriors": {
            "positive": posterior_positive,
            "negative": posterior_negative,
            "neutral": posterior_neutral
        },
        "update_steps": update_steps
    }

@register_reasoning_approach("vector_chain")
def vector_chain_approach(rules: List[Dict], facts: List[Dict], 
                         store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Vector-based reasoning chain approach that constructs coherent reasoning paths
    using hyperdimensional vector operations to connect facts to conclusions.
    
    Args:
        rules (list): List of rule dictionaries with vector representations
        facts (list): List of fact dictionaries with vector representations
        store (dict): Vector store for retrieving related vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary with chain parameters
        
    Returns:
        dict: Results dictionary containing:
              - outcome (str): Final recommendation based on reasoning chains
              - certainty (float): Confidence in the outcome (0.0-1.0)
              - conclusions (list): List of derived conclusions with vectors
              - chains (list): Structured reasoning chains with steps and certainty
    """
    logger.info("Applying vector chain reasoning approach")
    
    # Extract domain configuration
    domain_config = config.get("domain_config", {})
    positive_outcome = domain_config.get("positive_outcome", "POSITIVE")
    negative_outcome = domain_config.get("negative_outcome", "NEGATIVE")
    neutral_outcome = domain_config.get("neutral_outcome", "NEUTRAL")
    
    # Get similarity threshold from config
    similarity_threshold = config.get("similarity_threshold", 0.7)
    max_chain_depth = config.get("max_reasoning_depth", 5)
    
    # Get entity ID from facts if available
    entity_id = None
    if facts and "attributes" in facts[0]:
        entity_id = facts[0]["attributes"].get("entity_id", "")
    
    # Initialize tracking variables
    all_conclusions = []  # All derived conclusions
    intermediate_concepts = facts.copy()  # Start with facts as initial concepts
    chains = []  # To store reasoning chains
    
    # Keep track of derived concepts to avoid cycles
    derived_concept_ids = set()
    for fact in facts:
        derived_concept_ids.add(fact.get("identifier", ""))
    
    # Process in iterations to build chains of reasoning
    for iteration in range(max_chain_depth):
        logger.info(f"Chain reasoning iteration {iteration+1}/{max_chain_depth}")
        new_conclusions = []
        
        # Try to apply each rule to each concept in our current context
        for rule_idx, rule in enumerate(rules):
            rule_id = rule.get("identifier", f"rule_{rule_idx}")
            
            if not is_conditional(rule) or "vector" not in rule:
                continue
                
            rule_vector = rule["vector"]
            
            # Try to match the rule against each current concept
            for concept in intermediate_concepts:
                concept_id = concept.get("identifier", "")
                
                # Skip already derived concepts for this rule to avoid cycles
                derivation_key = f"{rule_id}_{concept_id}"
                if derivation_key in derived_concept_ids:
                    continue
                
                # Skip concepts without vectors
                if "vector" not in concept:
                    continue
                    
                concept_vector = concept["vector"]
                
                # Try to match the concept against the rule's antecedent
                try:
                    antecedent_text = extract_antecedent(rule)
                    match_result, similarity = matches(concept, antecedent_text, store)
                    
                    if match_result:
                        logger.info(f"Chain match: Rule {rule_id} matches concept {concept_id} with similarity {similarity:.4f}")
                        
                        try:
                            # Extract consequent for the conclusion
                            consequent_text = extract_consequent(rule)
                            
                            # Create a vector for the conclusion using vector operations
                            conclusion_vector = unbind_vectors(rule_vector, concept_vector)
                            conclusion_vector = cleanse_vector(conclusion_vector)
                            
                            # Create a unique identifier for the conclusion
                            conclusion_id = f"conclusion_{entity_id}_{len(all_conclusions)+1}"
                            
                            # Calculate certainty
                            rule_certainty = rule.get("attributes", {}).get("certainty", 0.9)
                            concept_certainty = concept.get("attributes", {}).get("certainty", 0.9)
                            match_certainty = similarity
                            certainty = min(rule_certainty, concept_certainty) * match_certainty
                            
                            # Create the conclusion
                            conclusion = {
                                "identifier": conclusion_id,
                                "type": "concept",
                                "vector": conclusion_vector,
                                "attributes": {
                                    "derived_from": [rule_id, concept_id],
                                    "derivation_method": "vector_chain",
                                    "rule_text": rule.get("attributes", {}).get("rule_text", ""),
                                    "concept_text": concept.get("attributes", {}).get("text", ""),
                                    "entity_id": entity_id,
                                    "certainty": certainty,
                                    "text": consequent_text,
                                    "chain_depth": iteration + 1
                                }
                            }
                            
                            # Classify the conclusion
                            signal_type = classify_signal_type(conclusion, domain_config)
                            conclusion["attributes"]["signal_type"] = signal_type
                            
                            # Add to new conclusions
                            new_conclusions.append(conclusion)
                            derived_concept_ids.add(derivation_key)
                            
                            # Record the reasoning step in a chain
                            chain_step = {
                                "step_number": iteration + 1,
                                "rule_id": rule_id,
                                "concept_id": concept_id,
                                "conclusion_id": conclusion_id,
                                "certainty": certainty,
                                "similarity": similarity,
                                "signal_type": signal_type
                            }
                            
                            # Find or create a chain for this branch
                            chain_found = False
                            for chain in chains:
                                # If this concept is the last conclusion in a chain, extend it
                                last_step = chain["steps"][-1]
                                if last_step["conclusion_id"] == concept_id:
                                    chain["steps"].append(chain_step)
                                    chain["final_conclusion_id"] = conclusion_id
                                    chain["depth"] = iteration + 1
                                    chain["final_certainty"] = certainty
                                    chain["signal_type"] = signal_type
                                    chain_found = True
                                    break
                            
                            # If not extending an existing chain, start a new one
                            if not chain_found:
                                # Check if this started from a fact
                                start_from_fact = concept_id in [f.get("identifier", "") for f in facts]
                                if start_from_fact:
                                    new_chain = {
                                        "chain_id": f"chain_{len(chains)+1}",
                                        "steps": [chain_step],
                                        "final_conclusion_id": conclusion_id,
                                        "depth": iteration + 1,
                                        "final_certainty": certainty,
                                        "signal_type": signal_type
                                    }
                                    chains.append(new_chain)
                            
                        except Exception as e:
                            logger.error(f"Error in chain reasoning: {str(e)}")
                
                except Exception as e:
                    logger.warning(f"Error checking for match: {str(e)}")
        
        # If no new conclusions were generated, we've reached a fixed point
        if not new_conclusions:
            logger.info(f"Chain reasoning reached fixed point after {iteration+1} iterations")
            break
            
        # Add new conclusions to the pool of intermediate concepts for next iteration
        intermediate_concepts.extend(new_conclusions)
        all_conclusions.extend(new_conclusions)
    
    # Calculate final results based on the chains
    if not chains:
        logger.info("No reasoning chains developed, defaulting to neutral outcome")
        return {
            "outcome": neutral_outcome,
            "certainty": 0.5,
            "conclusions": all_conclusions,
            "chains": [],
            "entity_id": entity_id
        }
    
    # Count strength of evidence from chains
    chain_weights = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    
    for chain in chains:
        signal_type = chain["signal_type"]
        weight = chain["final_certainty"]
        chain_weights[signal_type] += weight
    
    logger.info(f"Chain weights - Positive: {chain_weights['positive']:.4f}, "
               f"Negative: {chain_weights['negative']:.4f}, Neutral: {chain_weights['neutral']:.4f}")
    
    # Determine outcome
    total_weight = sum(chain_weights.values())
    
    if total_weight == 0:
        outcome = neutral_outcome
        certainty = 0.5
    elif chain_weights["positive"] > chain_weights["negative"]:
        outcome = positive_outcome
        certainty = 0.5 + (chain_weights["positive"] / total_weight) * 0.5
    elif chain_weights["negative"] > chain_weights["positive"]:
        outcome = negative_outcome
        certainty = 0.5 + (chain_weights["negative"] / total_weight) * 0.5
    else:
        outcome = neutral_outcome
        certainty = 0.5
    
    logger.info(f"Final outcome: {outcome} with certainty {certainty:.4f}")
    
    return {
        "outcome": outcome,
        "certainty": certainty,
        "conclusions": all_conclusions,
        "chains": chains,
        "entity_id": entity_id,
        "chain_weights": chain_weights
    }

# Legacy reasoning approaches (simplified versions that use the new vector operations)

@register_reasoning_approach("majority")
def majority_approach(rules: List[Dict], facts: List[Dict], 
                     store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Majority-based reasoning approach enhanced with vector operations.
    
    Args:
        rules (list): List of rule dictionaries with vector representations
        facts (list): List of fact dictionaries with vector representations
        store (dict): Vector store containing rule and fact vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary with domain settings
        
    Returns:
        dict: Results dictionary with outcome, certainty, and signal counts
    """
    logger.info("Applying enhanced majority reasoning approach")
    
    # Extract domain configuration
    domain_config = config.get("domain_config", {})
    positive_outcome = domain_config.get("positive_outcome", "POSITIVE")
    negative_outcome = domain_config.get("negative_outcome", "NEGATIVE")
    neutral_outcome = domain_config.get("neutral_outcome", "NEUTRAL")
    
    # Get entity ID from facts if available
    entity_id = None
    if facts and "attributes" in facts[0]:
        entity_id = facts[0]["attributes"].get("entity_id", "")
    
    # Initialize counters and storage
    positive_signals = 0
    negative_signals = 0
    neutral_signals = 0
    conclusions = []
    
    # Process rules and facts to generate conclusions
    for rule in rules:
        if is_conditional(rule):
            antecedent = extract_antecedent(rule)
            logger.debug(f"Checking rule with antecedent: {antecedent}")
            
            for fact in facts:
                match_result, similarity = matches(fact, antecedent, store)
                
                if match_result:
                    logger.debug(f"Matched fact: {fact.get('identifier', '')}")
                    
                    try:
                        # Extract consequent for the conclusion
                        consequent_text = extract_consequent(rule)
                        vector_dimension = config.get("vector_dimension", 10000)
                        
                        # Create a conclusion vector using vector operations if vectors available
                        if "vector" in rule and "vector" in fact:
                            conclusion_vector = unbind_vectors(rule["vector"], fact["vector"])
                            conclusion_vector = cleanse_vector(conclusion_vector)
                        else:
                            # Fallback: Generate a new vector
                            conclusion_vector = generate_vector(f"{rule.get('identifier')}_{fact.get('identifier')}", 
                                                              vector_dimension)
                        
                        # Create a unique identifier for the conclusion
                        conclusion_id = f"conclusion_{entity_id}_{len(conclusions)+1}"
                        
                        # Calculate certainty
                        rule_certainty = rule.get("attributes", {}).get("certainty", 0.9)
                        fact_certainty = fact.get("attributes", {}).get("certainty", 0.9)
                        certainty = min(rule_certainty, fact_certainty) * similarity
                        
                        # Create the conclusion
                        conclusion = {
                            "identifier": conclusion_id,
                            "type": "concept",
                            "vector": conclusion_vector,
                            "attributes": {
                                "derived_from": [rule.get("identifier"), fact.get("identifier")],
                                "derivation_method": "modus_ponens",
                                "rule_text": rule.get("attributes", {}).get("rule_text", ""),
                                "fact_text": fact.get("attributes", {}).get("fact_text", ""),
                                "entity_id": entity_id,
                                "certainty": certainty,
                                "text": consequent_text
                            }
                        }
                        
                        # Classify the conclusion
                        signal_type = classify_signal_type(conclusion, domain_config)
                        conclusion["attributes"]["signal_type"] = signal_type
                        
                        # Add to conclusions
                        conclusions.append(conclusion)
                        
                        # Update signal counters
                        if signal_type == "positive":
                            positive_signals += 1
                        elif signal_type == "negative":
                            negative_signals += 1
                        else:
                            neutral_signals += 1
                            
                    except Exception as e:
                        logger.error(f"Error generating conclusion: {str(e)}")
    
    # Determine outcome based on signal counts
    total_signals = positive_signals + negative_signals + neutral_signals
    logger.info(f"Signal counts - Positive: {positive_signals}, Negative: {negative_signals}, Neutral: {neutral_signals}")
    
    if total_signals == 0:
        logger.info("No signals found, defaulting to neutral outcome")
        return {
            "outcome": neutral_outcome,
            "certainty": 0.5,
            "conclusions": conclusions,
            "entity_id": entity_id,
            "signal_counts": {
                "positive": positive_signals,
                "negative": negative_signals,
                "neutral": neutral_signals
            }
        }
    
    # Calculate outcome and certainty
    if positive_signals > negative_signals:
        outcome = positive_outcome
        # Certainty scales from 0.5 to 1.0 based on ratio of positive signals to total
        certainty = 0.5 + ((positive_signals / total_signals) * 0.5)
        logger.info(f"Majority of positive signals, outcome: {outcome}, certainty: {certainty:.2f}")
    elif negative_signals > positive_signals:
        outcome = negative_outcome
        # Certainty scales from 0.5 to 1.0 based on ratio of negative signals to total
        certainty = 0.5 + ((negative_signals / total_signals) * 0.5)
        logger.info(f"Majority of negative signals, outcome: {outcome}, certainty: {certainty:.2f}")
    else:
        outcome = neutral_outcome
        certainty = 0.5
        logger.info(f"Equal signals, outcome: {outcome}, certainty: {certainty:.2f}")
    
    return {
        "outcome": outcome,
        "certainty": certainty,
        "conclusions": conclusions,
        "entity_id": entity_id,
        "signal_counts": {
            "positive": positive_signals,
            "negative": negative_signals,
            "neutral": neutral_signals
        }
    }


@register_reasoning_approach("weighted")
def weighted_approach(rules: List[Dict], facts: List[Dict], 
                     store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Weighted reasoning approach enhanced with vector operations.
    
    Args:
        rules (list): List of rule dictionaries with vector representations
        facts (list): List of fact dictionaries with vector representations
        store (dict): Vector store for retrieving related vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary with domain settings
        
    Returns:
        dict: Results dictionary with outcome, certainty, and evidence weights
    """
    logger.info("Applying enhanced weighted reasoning approach")
    
    # This is a bridge function that calls the more sophisticated vector_weighted_approach
    # to ensure backward compatibility while leveraging the improved vector operations
    return vector_weighted_approach(rules, facts, store, state, config)


@register_reasoning_approach("bayesian")
def bayesian_approach(rules: List[Dict], facts: List[Dict], 
                     store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Bayesian reasoning approach enhanced with vector operations.
    
    Args:
        rules (list): List of rule dictionaries with vector representations
        facts (list): List of fact dictionaries with vector representations
        store (dict): Vector store for retrieving related vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary with domain settings
        
    Returns:
        dict: Results dictionary with outcome, certainty, and posteriors
    """
    logger.info("Applying enhanced Bayesian reasoning approach")
    
    # This is a bridge function that calls the more sophisticated vector_bayesian_approach
    # to ensure backward compatibility while leveraging the improved vector operations
    return vector_bayesian_approach(rules, facts, store, state, config)
