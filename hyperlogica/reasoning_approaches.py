"""
Reasoning Approaches Module
===========================

This module implements domain-agnostic reasoning approaches for the Hyperlogica system.
Each approach applies different strategies for deriving conclusions from rules and facts.

The module follows functional programming principles with pure functions and explicit state passing.
"""

import re
import numpy as np
from typing import Dict, List, Any, Callable, Union, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


def is_conditional(rule: Dict) -> bool:
    """
    Check if a rule is a conditional statement.
    
    Args:
        rule (dict): Rule representation to check
        
    Returns:
        bool: True if the rule is conditional, False otherwise
    """
    # Check explicit attribute if available
    if "attributes" in rule and "conditional" in rule["attributes"]:
        return rule["attributes"]["conditional"]
    
    # Check if rule identifier contains "_if_" pattern
    if "identifier" in rule and "_if_" in rule["identifier"]:
        return True
    
    # Check if rule has antecedent and consequent attributes
    if ("attributes" in rule and 
        "antecedent" in rule["attributes"] and 
        "consequent" in rule["attributes"]):
        return True
    
    return False


def extract_antecedent(rule: Dict) -> str:
    """
    Extract the antecedent (condition) from a conditional rule.
    
    Args:
        rule (dict): Conditional rule representation
        
    Returns:
        str: Antecedent text or empty string if not found
        
    Raises:
        ValueError: If the rule is not a conditional
    """
    if not is_conditional(rule):
        raise ValueError("Cannot extract antecedent from non-conditional rule")
    
    # Try to get from attributes
    if "attributes" in rule and "antecedent" in rule["attributes"]:
        return rule["attributes"]["antecedent"]
    
    # Try to extract from identifier
    if "identifier" in rule and "_if_" in rule["identifier"]:
        parts = rule["identifier"].split("_if_")
        if len(parts) > 1:
            return parts[1]
    
    # Return empty string as fallback
    return ""


def extract_consequent(rule: Dict) -> str:
    """
    Extract the consequent (result) from a conditional rule.
    
    Args:
        rule (dict): Conditional rule representation
        
    Returns:
        str: Consequent text or empty string if not found
        
    Raises:
        ValueError: If the rule is not a conditional
    """
    if not is_conditional(rule):
        raise ValueError("Cannot extract consequent from non-conditional rule")
    
    # Try to get from attributes
    if "attributes" in rule and "consequent" in rule["attributes"]:
        return rule["attributes"]["consequent"]
    
    # Try to extract from identifier
    if "identifier" in rule and "_if_" in rule["identifier"]:
        parts = rule["identifier"].split("_if_")
        if len(parts) > 0:
            return parts[0]
    
    # Return empty string as fallback
    return ""


def matches(fact: Dict, antecedent: str, store: Dict) -> bool:
    """
    Check if a fact matches a rule's antecedent.
    
    Args:
        fact (dict): Fact representation to check
        antecedent (str): Antecedent text to match against
        store (dict): Vector store for similarity comparisons
        
    Returns:
        bool: True if the fact matches the antecedent, False otherwise
    """
    # If we have vector representations, use vector similarity
    if ("vector" in fact and "identifier" in fact and 
        store and "index" in store):
        # Get antecedent vector (would need to be implemented)
        # This is a simplified placeholder
        return False
    
    # Otherwise, perform text-based matching
    fact_text = ""
    if "attributes" in fact and "fact_text" in fact["attributes"]:
        fact_text = fact["attributes"]["fact_text"].lower()
    elif "identifier" in fact:
        fact_text = fact["identifier"].lower()
    
    antecedent_lower = antecedent.lower()
    
    # Simple text matching - in a real system this would be more sophisticated
    return antecedent_lower in fact_text


def apply_modus_ponens(rule: Dict, fact: Dict, store: Dict) -> Dict:
    """
    Apply modus ponens: If P→Q and P, then Q.
    
    Args:
        rule (dict): Conditional rule representation (P→Q)
        fact (dict): Fact representation matching the antecedent (P)
        store (dict): Vector store for retrieving related vectors
        
    Returns:
        dict: Derived conclusion representation (Q) with certainty
        
    Raises:
        ValueError: If the rule is not a conditional
    """
    if not is_conditional(rule):
        raise ValueError("Cannot apply modus ponens to non-conditional rule")
    
    # Extract consequent
    consequent_text = extract_consequent(rule)
    
    # Generate identifier for the conclusion
    conclusion_id = f"derived_{rule.get('identifier', 'rule')}_{fact.get('identifier', 'fact')}"
    
    # Calculate certainty: min(certainty(P→Q), certainty(P))
    rule_certainty = rule.get("certainty", 1.0)
    fact_certainty = fact.get("certainty", 1.0)
    certainty = min(rule_certainty, fact_certainty)
    
    # Create conclusion representation
    conclusion = {
        "identifier": conclusion_id,
        "certainty": certainty,
        "attributes": {
            "derived_from": [rule.get("identifier", ""), fact.get("identifier", "")],
            "derived_method": "modus_ponens",
            "text": consequent_text
        }
    }
    
    # If vectors are available, we would compute the conclusion vector here
    # This is a simplified placeholder
    
    return conclusion


def is_signal_type(conclusion: Dict, signal_type: str, domain_config: Dict) -> bool:
    """
    Check if a conclusion represents a specific signal type (positive, negative, neutral).
    
    Args:
        conclusion (dict): Conclusion representation to check
        signal_type (str): Signal type to check for ("positive", "negative", "neutral")
        domain_config (dict): Domain-specific configuration containing signal keywords
        
    Returns:
        bool: True if the conclusion matches the signal type, False otherwise
    """
    # Get the appropriate keywords for the signal type
    keywords = []
    if signal_type == "positive":
        keywords = domain_config.get("positive_outcome_keywords", [])
    elif signal_type == "negative":
        keywords = domain_config.get("negative_outcome_keywords", [])
    elif signal_type == "neutral":
        keywords = domain_config.get("neutral_outcome_keywords", [])
    
    # Search for keywords in the conclusion
    conclusion_text = ""
    if "attributes" in conclusion and "text" in conclusion["attributes"]:
        conclusion_text = conclusion["attributes"]["text"].lower()
    elif "identifier" in conclusion:
        conclusion_text = conclusion["identifier"].lower()
    
    # Check if any keywords match
    for keyword in keywords:
        if keyword.lower() in conclusion_text:
            return True
    
    return False


def extract_likelihood_data(conclusion: Dict, rule: Dict, domain_config: Dict) -> Dict[str, float]:
    """
    Extract likelihood values for each possible outcome from a conclusion.
    
    Args:
        conclusion (dict): Conclusion representation to analyze
        rule (dict): Rule that generated the conclusion
        domain_config (dict): Domain-specific configuration 
        
    Returns:
        dict: Dictionary with likelihood values for each outcome type
    """
    # Start with default uniform likelihoods
    likelihood = {
        "positive": 0.5,
        "negative": 0.5,
        "neutral": 0.5
    }
    
    # Adjust based on conclusion certainty
    certainty = conclusion.get("certainty", 0.5)
    
    # Check which signal type the conclusion represents
    if is_signal_type(conclusion, "positive", domain_config):
        likelihood["positive"] = certainty
        likelihood["negative"] = 1.0 - certainty
        likelihood["neutral"] = 0.5
    elif is_signal_type(conclusion, "negative", domain_config):
        likelihood["positive"] = 1.0 - certainty
        likelihood["negative"] = certainty
        likelihood["neutral"] = 0.5
    else:
        likelihood["neutral"] = certainty
    
    return likelihood


@register_reasoning_approach("majority")
def majority_approach(rules: List[Dict], facts: List[Dict], 
                     store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Generic majority-based reasoning approach that decides based on the count of
    positive vs. negative signals.
    
    Args:
        rules (list): List of processed rule representations
        facts (list): List of processed fact representations
        store (dict): Vector store containing rule and fact vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary containing reasoning settings
                      and domain-specific parameters
        
    Returns:
        dict: Results dictionary containing:
              - outcome (str): Final recommendation based on majority voting
              - certainty (float): Confidence in the outcome (0.5-1.0)
              - conclusions (list): List of derived conclusions
              - signal_counts (dict): Counts of positive, negative, and neutral signals
    """
    logger.info("Applying majority reasoning approach")
    
    # Extract domain configuration
    domain_config = config.get("processing", {}).get("domain_config", {})
    positive_outcome = domain_config.get("positive_outcome", "POSITIVE")
    negative_outcome = domain_config.get("negative_outcome", "NEGATIVE")
    neutral_outcome = domain_config.get("neutral_outcome", "NEUTRAL")
    
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
                if matches(fact, antecedent, store):
                    logger.debug(f"Matched fact: {fact.get('identifier', '')}")
                    conclusion = apply_modus_ponens(rule, fact, store)
                    conclusions.append(conclusion)
                    
                    # Classify the conclusion based on domain config
                    if is_signal_type(conclusion, "positive", domain_config):
                        positive_signals += 1
                        logger.debug(f"Found positive signal: {conclusion.get('identifier', '')}")
                    elif is_signal_type(conclusion, "negative", domain_config):
                        negative_signals += 1
                        logger.debug(f"Found negative signal: {conclusion.get('identifier', '')}")
                    else:
                        neutral_signals += 1
                        logger.debug(f"Found neutral signal: {conclusion.get('identifier', '')}")
    
    # Determine outcome based on signal counts
    total_signals = positive_signals + negative_signals + neutral_signals
    logger.info(f"Signal counts - Positive: {positive_signals}, Negative: {negative_signals}, Neutral: {neutral_signals}")
    
    if total_signals == 0:
        logger.info("No signals found, defaulting to neutral outcome")
        return {
            "outcome": neutral_outcome,
            "certainty": 0.5,
            "conclusions": conclusions,
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
        "signal_counts": {
            "positive": positive_signals,
            "negative": negative_signals,
            "neutral": neutral_signals
        }
    }


@register_reasoning_approach("weighted")
def weighted_approach(rules: list, facts: list, store: dict, state: dict, config: dict) -> dict:
    """
    Apply a weighted evidence reasoning approach using vector similarity for matching rules to facts.
    
    This function implements a reasoning strategy that:
    1. Extracts antecedents and consequents from rules
    2. Uses cosine similarity to match rule conditions to facts
    3. Generates conclusions when similarities exceed a configurable threshold
    4. Classifies evidence as positive, negative, or neutral based on keywords
    5. Weighs evidence to determine an overall recommendation
    
    Args:
        rules (list): List of rule dictionaries, each containing vector representations and attributes
        facts (list): List of fact dictionaries, each containing vector representations and attributes
        store (dict): Vector store containing additional information about rules and facts
        state (dict): Current state dictionary for tracking context
        config (dict): Configuration dictionary containing reasoning parameters, including:
                      - similarity_threshold: Minimum similarity score to consider a match (default: 0.5)
                      - domain_config: Dictionary with domain-specific settings
    
    Returns:
        dict: Dictionary containing reasoning results with the following keys:
              - outcome (str): Final recommendation (e.g., "BUY", "SELL", "HOLD")
              - certainty (float): Confidence level in the recommendation (0.5-1.0)
              - conclusions (list): List of derived conclusions with their sources
              - evidence_weights (dict): Breakdown of positive, negative, and neutral evidence weights
    """
    logging.info("Applying weighted reasoning approach with vector similarity")
    
    domain_config = config.get("domain_config", {})
    positive_outcome = domain_config.get("positive_outcome", "BUY")
    negative_outcome = domain_config.get("negative_outcome", "SELL")
    neutral_outcome = domain_config.get("neutral_outcome", "HOLD")
    
    # Lower similarity threshold
    similarity_threshold = config.get("similarity_threshold", 0.5)
    logging.info(f"Using similarity threshold: {similarity_threshold}")
    
    positive_evidence = 0.0
    negative_evidence = 0.0
    neutral_evidence = 0.0
    conclusions = []
    
    # Process rules and facts with weighting and vector similarity
    for rule in rules:
        rule_id = rule.get('identifier', 'unknown')
        logging.info(f"Evaluating rule: {rule_id}")
        
        if 'attributes' not in rule or not rule.get('vector', None) is not None:
            logging.warning(f"Rule missing attributes or vector: {rule_id}")
            continue
            
        # Check if this is a conditional rule
        rule_text = rule.get('attributes', {}).get('rule_text', '')
        logging.info(f"Rule text: {rule_text}")
        
        # Extract antecedent - from attributes or from text
        antecedent = rule.get('attributes', {}).get('antecedent', '')
        if not antecedent and 'if' in rule_text.lower() and 'then' in rule_text.lower():
            parts = rule_text.lower().split('then')
            antecedent_part = parts[0]
            if antecedent_part.startswith('if '):
                antecedent = antecedent_part[3:].strip()
        
        logging.info(f"Rule antecedent: {antecedent}")
        
        # Extract consequent from attributes or from text
        consequent = rule.get('attributes', {}).get('consequent', '')
        if not consequent and 'if' in rule_text.lower() and 'then' in rule_text.lower():
            parts = rule_text.lower().split('then')
            if len(parts) > 1:
                consequent = parts[1].strip()
        
        logging.info(f"Rule consequent: {consequent}")
        
        # Only proceed if we have a conditional rule
        is_conditional = ('if' in rule_text.lower() and 'then' in rule_text.lower()) or rule.get('attributes', {}).get('conditional', False)
        
        if is_conditional:
            rule_vector = rule.get('vector')
            
            # Check each fact for similarity to this rule's antecedent
            for fact in facts:
                fact_id = fact.get('identifier', 'unknown')
                if 'vector' not in fact:
                    logging.warning(f"Fact missing vector: {fact_id}")
                    continue
                
                fact_text = fact.get('attributes', {}).get('fact_text', '')
                logging.info(f"Comparing to fact: {fact_id}: {fact_text}")
                
                fact_vector = fact.get('vector')
                
                # Calculate vector similarity using cosine similarity
                dot_product = np.dot(rule_vector, fact_vector)
                rule_norm = np.linalg.norm(rule_vector)
                fact_norm = np.linalg.norm(fact_vector)
                
                if rule_norm == 0 or fact_norm == 0:
                    similarity = 0
                else:
                    similarity = dot_product / (rule_norm * fact_norm)
                
                logging.info(f"Similarity between rule {rule_id} and fact {fact_id}: {similarity:.4f}")
                
                # If similarity exceeds threshold, consider it a match
                if similarity >= similarity_threshold:
                    logging.info(f"Match found! Rule {rule_id} matches fact {fact_id}")
                    
                    # Create a conclusion based on this match
                    if not consequent:
                        consequent = "Derived conclusion from rule"
                    
                    conclusion = {
                        "identifier": f"conclusion_{len(conclusions)+1}",
                        "source_rule": rule_id,
                        "source_fact": fact_id,
                        "text": consequent,
                        "similarity": similarity,
                        "attributes": {
                            "certainty": min(rule.get('attributes', {}).get('certainty', 0.8), 
                                         fact.get('attributes', {}).get('certainty', 0.8)) * similarity
                        }
                    }
                    
                    conclusions.append(conclusion)
                    
                    # Determine if this is a positive, negative, or neutral signal
                    conclusion_text = consequent.lower()
                    
                    # Check if it contains positive/negative keywords
                    if any(keyword in conclusion_text for keyword in domain_config.get("positive_outcome_keywords", [])):
                        signal_type = "positive"
                        weight = conclusion["attributes"]["certainty"]
                        positive_evidence += weight
                        logging.info(f"Added positive evidence: {weight:.4f}")
                    elif any(keyword in conclusion_text for keyword in domain_config.get("negative_outcome_keywords", [])):
                        signal_type = "negative"
                        weight = conclusion["attributes"]["certainty"]
                        negative_evidence += weight
                        logging.info(f"Added negative evidence: {weight:.4f}")
                    else:
                        signal_type = "neutral"
                        weight = conclusion["attributes"]["certainty"]
                        neutral_evidence += weight
                        logging.info(f"Added neutral evidence: {weight:.4f}")
                    
                    # Add signal type to conclusion
                    conclusion["attributes"]["signal_type"] = signal_type
    
    # Log the total evidence found
    logging.info(f"Evidence weights - Positive: {positive_evidence:.2f}, Negative: {negative_evidence:.2f}, Neutral: {neutral_evidence:.2f}")
    
    # Determine outcome based on weighted evidence
    total_evidence = positive_evidence + negative_evidence + neutral_evidence
    
    if total_evidence == 0:
        logging.info("No evidence found, defaulting to neutral outcome")
        return {
            "outcome": neutral_outcome,
            "certainty": 0.5,
            "conclusions": conclusions,
            "evidence_weights": {
                "positive": positive_evidence,
                "negative": negative_evidence,
                "neutral": neutral_evidence
            }
        }
    
    if positive_evidence > negative_evidence:
        outcome = positive_outcome
        certainty = 0.5 + (positive_evidence / total_evidence) * 0.5
        logging.info(f"Positive evidence dominates: {positive_evidence:.2f} > {negative_evidence:.2f}, certainty: {certainty:.2f}")
    elif negative_evidence > positive_evidence:
        outcome = negative_outcome
        certainty = 0.5 + (negative_evidence / total_evidence) * 0.5
        logging.info(f"Negative evidence dominates: {negative_evidence:.2f} > {positive_evidence:.2f}, certainty: {certainty:.2f}")
    else:
        outcome = neutral_outcome
        certainty = 0.5
        logging.info("Evidence is balanced, neutral outcome")
    
    return {
        "outcome": outcome,
        "certainty": certainty,
        "conclusions": conclusions,
        "evidence_weights": {
            "positive": positive_evidence,
            "negative": negative_evidence,
            "neutral": neutral_evidence
        }
    }


@register_reasoning_approach("bayesian")
def bayesian_approach(rules: List[Dict], facts: List[Dict], 
                     store: Dict, state: Dict, config: Dict) -> Dict:
    """
    Generic Bayesian reasoning approach that updates posterior probabilities
    of each outcome type based on the evidence.
    
    Args:
        rules (list): List of processed rule representations
        facts (list): List of processed fact representations
        store (dict): Vector store containing rule and fact vectors
        state (dict): State dictionary for tracking reasoning context
        config (dict): Configuration dictionary containing reasoning settings
                      and domain-specific parameters
        
    Returns:
        dict: Results dictionary containing:
              - outcome (str): Final recommendation based on Bayesian reasoning
              - certainty (float): Confidence in the outcome (0.0-1.0)
              - conclusions (list): List of derived conclusions
              - posteriors (dict): Posterior probabilities for each possible outcome
    """
    logger.info("Applying Bayesian reasoning approach")
    
    # Extract domain configuration
    domain_config = config.get("processing", {}).get("domain_config", {})
    positive_outcome = domain_config.get("positive_outcome", "POSITIVE")
    negative_outcome = domain_config.get("negative_outcome", "NEGATIVE")
    neutral_outcome = domain_config.get("neutral_outcome", "NEUTRAL")
    
    # Initial priors (equal by default, but configurable)
    prior_positive = domain_config.get("prior_positive", 1/3)
    prior_negative = domain_config.get("prior_negative", 1/3)
    prior_neutral = domain_config.get("prior_neutral", 1/3)
    
    logger.debug(f"Initial priors - Positive: {prior_positive:.2f}, Negative: {prior_negative:.2f}, Neutral: {prior_neutral:.2f}")
    
    # Normalize priors to ensure they sum to 1.0
    prior_sum = prior_positive + prior_negative + prior_neutral
    if prior_sum > 0:
        prior_positive /= prior_sum
        prior_negative /= prior_sum
        prior_neutral /= prior_sum
    else:
        # Default to equal priors if sum is 0
        prior_positive = prior_negative = prior_neutral = 1/3
    
    conclusions = []
    likelihood_data = []
    update_steps = []
    
    # Process rules and facts to generate conclusions
    for rule in rules:
        if is_conditional(rule):
            antecedent = extract_antecedent(rule)
            logger.debug(f"Checking rule with antecedent: {antecedent}")
            
            for fact in facts:
                if matches(fact, antecedent, store):
                    logger.debug(f"Matched fact: {fact.get('identifier', '')}")
                    conclusion = apply_modus_ponens(rule, fact, store)
                    conclusions.append(conclusion)
                    
                    # Extract likelihood data for this conclusion
                    likelihood = extract_likelihood_data(conclusion, rule, domain_config)
                    likelihood_data.append(likelihood)
                    logger.debug(f"Extracted likelihood: {likelihood}")
    
    # Update posterior probabilities using Bayes' theorem
    posterior_positive = prior_positive
    posterior_negative = prior_negative
    posterior_neutral = prior_neutral
    
    # Process each piece of evidence sequentially
    for idx, likelihood in enumerate(likelihood_data):
        # Record the current state before update
        pre_update = {
            "step": idx + 1,
            "prior_positive": posterior_positive,
            "prior_negative": posterior_negative,
            "prior_neutral": posterior_neutral,
            "likelihood": likelihood
        }
        
        # Calculate the denominator for Bayes' theorem
        denominator = (
            posterior_positive * likelihood.get("positive", 0.5) + 
            posterior_negative * likelihood.get("negative", 0.5) + 
            posterior_neutral * likelihood.get("neutral", 0.5)
        )
        
        # Update posteriors if denominator is valid
        if denominator > 0:
            new_posterior_positive = (posterior_positive * likelihood.get("positive", 0.5)) / denominator
            new_posterior_negative = (posterior_negative * likelihood.get("negative", 0.5)) / denominator
            new_posterior_neutral = (posterior_neutral * likelihood.get("neutral", 0.5)) / denominator
            
            posterior_positive = new_posterior_positive
            posterior_negative = new_posterior_negative
            posterior_neutral = new_posterior_neutral
            
            logger.debug(f"Updated posteriors - Positive: {posterior_positive:.2f}, "
                        f"Negative: {posterior_negative:.2f}, Neutral: {posterior_neutral:.2f}")
        else:
            logger.warning(f"Skipping Bayesian update step {idx+1} due to zero denominator")
        
        # Record the posteriors after update
        post_update = {
            "posterior_positive": posterior_positive,
            "posterior_negative": posterior_negative,
            "posterior_neutral": posterior_neutral
        }
        
        update_steps.append({**pre_update, **post_update})
    
    # Determine outcome based on posterior probabilities
    if posterior_positive > posterior_negative and posterior_positive > posterior_neutral:
        outcome = positive_outcome
        certainty = posterior_positive
        logger.info(f"Highest posterior for positive outcome: {outcome}, certainty: {certainty:.2f}")
    elif posterior_negative > posterior_positive and posterior_negative > posterior_neutral:
        outcome = negative_outcome
        certainty = posterior_negative
        logger.info(f"Highest posterior for negative outcome: {outcome}, certainty: {certainty:.2f}")
    else:
        outcome = neutral_outcome
        certainty = posterior_neutral
        logger.info(f"Highest posterior for neutral outcome: {outcome}, certainty: {certainty:.2f}")
    
    return {
        "outcome": outcome,
        "certainty": certainty,
        "conclusions": conclusions,
        "posteriors": {
            "positive": posterior_positive,
            "negative": posterior_negative,
            "neutral": posterior_neutral
        },
        "update_steps": update_steps
    }
