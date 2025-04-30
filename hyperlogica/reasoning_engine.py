"""
Reasoning Engine Module

Pure functional implementation of the hyperdimensional reasoning engine for Hyperlogica,
focused on vector-chain reasoning with proper ACEP representation.
"""

import logging
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from datetime import datetime

# Import vector operations
from .vector_operations import (
    normalize_vector, bind_vectors, unbind_vectors, bundle_vectors, 
    calculate_similarity, cleanse_vector, create_role_vectors
)

# Configure logging
logger = logging.getLogger(__name__)

def match_condition_to_fact(rule: Dict[str, Any], fact: Dict[str, Any], 
                          roles: Dict[str, np.ndarray],
                          similarity_threshold: float = 0.7) -> Tuple[bool, float]:
    """
    Check if a fact matches a rule's condition using vector similarity.
    
    Args:
        rule (Dict[str, Any]): Rule representation with condition vector
        fact (Dict[str, Any]): Fact representation with vector
        roles (Dict[str, np.ndarray]): Role vectors for structure
        similarity_threshold (float): Threshold for considering a match
        
    Returns:
        Tuple[bool, float]: (match_result, similarity_score)
    """
    # Get vectors - both should be tagged with condition_role for structural compatibility
    if "condition_vector" not in rule:
        logger.warning(f"Rule {rule.get('identifier', 'unknown')} has no condition_vector")
        return False, 0.0
        
    if "vector" not in fact:
        logger.warning(f"Fact {fact.get('identifier', 'unknown')} has no vector")
        return False, 0.0
    
    condition_vector = rule["condition_vector"]
    fact_vector = fact["vector"]
    
    # Log vectors for debugging (sample values)
    if logger.isEnabledFor(logging.DEBUG):
        # Log just a small sample of the vectors to avoid excessive output
        sample_size = min(5, len(condition_vector))
        logger.debug(f"Condition vector sample: {condition_vector[:sample_size]}")
        logger.debug(f"Fact vector sample: {fact_vector[:sample_size]}")
    
    # Check for NaN or infinite values
    if np.isnan(condition_vector).any() or np.isinf(condition_vector).any():
        logger.warning("Condition vector contains NaN or Inf values")
        condition_vector = np.nan_to_num(condition_vector)
    
    if np.isnan(fact_vector).any() or np.isinf(fact_vector).any():
        logger.warning("Fact vector contains NaN or Inf values")
        fact_vector = np.nan_to_num(fact_vector)
    
    # Calculate similarity - these should now be structurally comparable
    similarity = calculate_similarity(condition_vector, fact_vector)
    
    # Log ACEP content for high-similarity matches
    if similarity > 0.5:
        rule_condition = rule.get("acep", {}).get("content", {}).get("condition", {})
        fact_content = fact.get("acep", {}).get("content", {})
        
        rule_concept = rule_condition.get("concept", "")
        fact_concept = fact_content.get("concept", "")
        
        rule_relation = rule_condition.get("relation", "")
        fact_relation = fact_content.get("relation", "")
        
        # Log the concepts and relations being compared
        logger.debug(f"High similarity ({similarity:.4f}): {rule_concept}_{rule_relation} vs {fact_concept}_{fact_relation}")
    
    # Check if similarity exceeds threshold
    if similarity >= similarity_threshold:
        logger.info(f"Match found! Similarity: {similarity:.4f}")
        
        # Log details of the match
        rule_id = rule.get("identifier", "unknown")
        fact_id = fact.get("identifier", "unknown")
        logger.info(f"Rule {rule_id} matched fact {fact_id} with similarity {similarity:.4f}")
        
        rule_condition = rule.get("acep", {}).get("content", {}).get("condition", {})
        fact_content = fact.get("acep", {}).get("content", {})
        logger.info(f"Rule condition: {rule_condition.get('concept', '')}-{rule_condition.get('relation', '')}-{rule_condition.get('reference', '')}")
        logger.info(f"Fact content: {fact_content.get('concept', '')}-{fact_content.get('relation', '')}-{fact_content.get('reference', '')}")
        
        return True, similarity
    
    return False, similarity


def create_conclusion(rule: Dict[str, Any], fact: Dict[str, Any], 
                    similarity: float, entity_id: str,
                    roles: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Create a conclusion by applying a rule to a fact.
    
    Args:
        rule (Dict[str, Any]): Rule that matched
        fact (Dict[str, Any]): Fact that matched the rule's condition
        similarity (float): Similarity score of the match
        entity_id (str): Entity ID this conclusion applies to
        roles (Dict[str, np.ndarray]): Role vectors for unbinding
        
    Returns:
        Dict[str, Any]: The derived conclusion with its vector
    """
    # Create a unique ID for the conclusion
    conclusion_id = str(uuid.uuid4())[:8]
    if entity_id:
        conclusion_id = f"{entity_id}_conclusion_{conclusion_id}"
    
    # Extract implication vector
    implication_vector = rule.get("implication_vector")
    if implication_vector is None:
        logger.warning("No implication vector found in rule, using fallback")
        # Use fallback - construct from original vectors if available
        rule_vector = rule.get("vector")
        if rule_vector is not None and "condition_vector" in rule:
            # Try to unbind to get implication
            implication_vector = unbind_vectors(rule_vector, rule["condition_vector"])
            implication_vector = cleanse_vector(implication_vector)
        else:
            # Create a random vector as last resort
            dimension = fact["vector"].shape[0]
            implication_vector = np.random.normal(0, 1, dimension)
            implication_vector = normalize_vector(implication_vector)
    
    # Calculate certainty for the conclusion
    rule_certainty = rule.get("attributes", {}).get("certainty", 0.9)
    fact_certainty = fact.get("attributes", {}).get("certainty", 0.8)
    
    # Certainty is the minimum of rule certainty, fact certainty, and match similarity
    conclusion_certainty = min(rule_certainty, fact_certainty) * similarity
    
    # Extract the ACEP representation for the implication
    implication_acep = rule.get("acep", {}).get("content", {}).get("implication", {})
    
    # Create ACEP representation for the conclusion
    conclusion_acep = {
        "type": "derived_assertion",
        "identifier": conclusion_id,
        "content": {
            "concept": implication_acep.get("concept", ""),
            "state": implication_acep.get("state", ""),
            "derivation": "modus_ponens"
        },
        "attributes": {
            "certainty": conclusion_certainty,
            "entity_id": entity_id,
            "derived_from": [
                rule.get("identifier", "unknown_rule"),
                fact.get("identifier", "unknown_fact")
            ],
            "match_similarity": similarity,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Create the conclusion with its vector
    conclusion = {
        "identifier": conclusion_id,
        "vector": implication_vector,
        "acep": conclusion_acep,
        "condition_vector": None,
        "implication_vector": None,
        "attributes": conclusion_acep["attributes"],
    }
    
    return conclusion

def apply_vector_chain_reasoning(rules: List[Dict[str, Any]], facts: List[Dict[str, Any]], 
                                entity_id: str, roles: Dict[str, np.ndarray],
                                max_depth: int = 5, 
                                similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Apply vector chain reasoning to derive conclusions through multiple steps.
    
    Args:
        rules (List[Dict[str, Any]]): Rules with ACEP representations and vectors
        facts (List[Dict[str, Any]]): Facts with ACEP representations and vectors
        entity_id (str): Entity ID these facts belong to
        roles (Dict[str, np.ndarray]): Role vectors for binding/unbinding
        max_depth (int): Maximum reasoning depth
        similarity_threshold (float): Threshold for considering a match
        
    Returns:
        Dict[str, Any]: Reasoning results including chains and conclusions
    """
    # Initialize reasoning state
    depth = 0
    current_facts = facts.copy()  # Start with initial facts
    all_conclusions = []  # All derived conclusions
    chains = []  # Reasoning chains
    derived_concept_ids = set()  # Track derived concepts to avoid cycles
    
    # Track positive, negative, and neutral conclusions
    positive_conclusions = []
    negative_conclusions = []
    neutral_conclusions = []
    
    # Process each fact to extract its entity ID if needed
    if not entity_id and facts:
        entity_id = facts[0].get("attributes", {}).get("entity_id", "")
    
    # Initial state
    logger.info(f"Starting vector chain reasoning for entity {entity_id}")
    logger.info(f"Initial facts: {len(facts)}, Rules: {len(rules)}")
    
    # Enhanced debugging - log the first fact and rule concepts for inspection
    if facts and rules:
        first_fact = facts[0]
        first_rule = rules[0]
        fact_acep = first_fact.get("acep", {})
        rule_acep = first_rule.get("acep", {})
        
        logger.info(f"Sample fact concept: {fact_acep.get('content', {}).get('concept', 'N/A')}")
        logger.info(f"Sample rule condition concept: {rule_acep.get('content', {}).get('condition', {}).get('concept', 'N/A')}")
        
        # Log vector shapes and types
        if "vector" in first_fact and "condition_vector" in first_rule:
            logger.info(f"Fact vector shape: {first_fact['vector'].shape}, type: {first_fact['vector'].dtype}")
            logger.info(f"Rule condition vector shape: {first_rule['condition_vector'].shape}, type: {first_rule['condition_vector'].dtype}")
    
    # Pre-match similarity check for first rule-fact pair (diagnostic)
    if facts and rules:
        test_match, test_similarity = match_condition_to_fact(rules[0], facts[0], roles, 0.0)  # Use 0.0 threshold for testing
        logger.info(f"Test match similarity: {test_similarity:.4f} (threshold is {similarity_threshold:.4f})")
    
    # Reasoning iterations
    while depth < max_depth:
        logger.info(f"Reasoning depth: {depth+1}/{max_depth}")
        new_conclusions = []
        
        # Try to match each rule with each fact
        match_attempts = 0
        for rule in rules:
            rule_id = rule.get("identifier", f"rule_{uuid.uuid4()}")
            
            for fact in current_facts:
                fact_id = fact.get("identifier", f"fact_{uuid.uuid4()}")
                match_attempts += 1
                
                # Skip already derived concepts for this rule to avoid cycles
                derivation_key = f"{rule_id}_{fact_id}"
                if derivation_key in derived_concept_ids:
                    continue
                
                # Try to match the fact against the rule's condition
                match_result, similarity = match_condition_to_fact(rule, fact, roles, similarity_threshold)
                
                if match_result:
                    logger.info(f"Match found: Rule {rule_id} matches fact {fact_id} with similarity {similarity:.4f}")
                    
                    # Create a conclusion
                    conclusion = create_conclusion(rule, fact, similarity, entity_id, roles)
                    conclusion_id = conclusion.get("identifier", "")
                    
                    # Add to derived concepts to avoid cycles
                    derived_concept_ids.add(derivation_key)
                    
                    # Add to new conclusions
                    new_conclusions.append(conclusion)
                    all_conclusions.append(conclusion)
                    
                    # Create a chain step record
                    step = {
                        "step_number": depth + 1,
                        "rule_id": rule_id,
                        "fact_id": fact_id,
                        "conclusion_id": conclusion_id,
                        "similarity": similarity,
                        "certainty": conclusion.get("attributes", {}).get("certainty", 0.0),
                        "acep": conclusion.get("acep", {})
                    }
                    
                    # Check if this extends an existing chain
                    chain_extended = False
                    for chain in chains:
                        last_step = chain["steps"][-1]
                        if last_step["conclusion_id"] == fact_id:
                            # Extend this chain
                            chain["steps"].append(step)
                            chain["final_conclusion_id"] = conclusion_id
                            chain["depth"] = depth + 1
                            chain["final_certainty"] = step["certainty"]
                            chain_extended = True
                            break
                    
                    # If not extending a chain, start a new one
                    if not chain_extended:
                        # Check if this started from an original fact (not a derived conclusion)
                        if fact_id in [f.get("identifier", "") for f in facts]:
                            chain = {
                                "chain_id": f"chain_{len(chains)+1}",
                                "steps": [step],
                                "final_conclusion_id": conclusion_id,
                                "depth": 1,
                                "final_certainty": step["certainty"],
                                "entity_id": entity_id
                            }
                            chains.append(chain)
                    
                    # Classify the conclusion as positive, negative, or neutral
                    signal_type = classify_conclusion(conclusion)
                    
                    if signal_type == "positive":
                        positive_conclusions.append(conclusion)
                    elif signal_type == "negative":
                        negative_conclusions.append(conclusion)
                    else:
                        neutral_conclusions.append(conclusion)
        
        # Log match statistics
        logger.info(f"Attempted {match_attempts} rule-fact matches at depth {depth+1}")
        
        # If no new conclusions were generated, we've reached a fixed point
        if not new_conclusions:
            logger.info(f"Reasoning reached fixed point after {depth+1} iterations")
            break
        
        # Add new conclusions to facts for next iteration
        current_facts.extend(new_conclusions)
        depth += 1
    
    # Calculate final results
    total_positive = sum(c.get("attributes", {}).get("certainty", 0.0) for c in positive_conclusions)
    total_negative = sum(c.get("attributes", {}).get("certainty", 0.0) for c in negative_conclusions)
    total_neutral = sum(c.get("attributes", {}).get("certainty", 0.0) for c in neutral_conclusions)
    
    # Calculate total evidence
    total_evidence = total_positive + total_negative + total_neutral
    
    # Determine the outcome based on strongest evidence
    if total_evidence == 0:
        outcome = "NEUTRAL"
        certainty = 0.5
    elif total_positive > total_negative:
        outcome = "POSITIVE"
        certainty = 0.5 + (total_positive / total_evidence) * 0.5
    elif total_negative > total_positive:
        outcome = "NEGATIVE"
        certainty = 0.5 + (total_negative / total_evidence) * 0.5
    else:
        outcome = "NEUTRAL"
        certainty = 0.5
    
    logger.info(f"Reasoning complete: {outcome} with certainty {certainty:.4f}")
    logger.info(f"Derived {len(all_conclusions)} conclusions through {len(chains)} chains")
    
    # Prepare the reasoning result
    result = {
        "outcome": outcome,
        "certainty": certainty,
        "conclusions": all_conclusions,
        "chains": chains,
        "positive_conclusions": positive_conclusions,
        "negative_conclusions": negative_conclusions,
        "neutral_conclusions": neutral_conclusions,
        "evidence_weights": {
            "positive": total_positive,
            "negative": total_negative,
            "neutral": total_neutral
        },
        "entity_id": entity_id,
        "termination_reason": "fixed_point" if depth < max_depth else "max_depth_reached",
        "final_depth": depth
    }
    
    return result

def classify_conclusion(conclusion: Dict[str, Any]) -> str:
    """
    Classify a conclusion as positive, negative, or neutral based on its content.
    
    Args:
        conclusion (Dict[str, Any]): Conclusion to classify
        
    Returns:
        str: Signal type ("positive", "negative", "neutral")
    """
    # Default to neutral
    signal_type = "neutral"
    
    # Check if there's an explicit signal type or valence attribute
    attributes = conclusion.get("attributes", {})
    if "signal_type" in attributes:
        return attributes["signal_type"]
    
    if "valence" in attributes:
        valence = attributes["valence"].lower()
        if valence in ["positive", "good", "favorable"]:
            return "positive"
        elif valence in ["negative", "bad", "unfavorable"]:
            return "negative"
        return "neutral"
    
    # Check the implication state if available
    acep = conclusion.get("acep", {})
    content = acep.get("content", {})
    state = content.get("state", "").lower()
    
    # Classify based on state
    positive_states = ["undervalued", "improving", "increasing", "growing", "expanding", "positive", "buy"]
    negative_states = ["overvalued", "declining", "decreasing", "contracting", "negative", "sell"]
    
    if state in positive_states:
        signal_type = "positive"
    elif state in negative_states:
        signal_type = "negative"
    
    return signal_type

def generate_explanation(reasoning_result: Dict[str, Any]) -> str:
    """
    Generate a natural language explanation of the reasoning process.
    
    Args:
        reasoning_result (Dict[str, Any]): Result of the reasoning process
        
    Returns:
        str: Natural language explanation
    """
    # Extract key components from the reasoning result
    outcome = reasoning_result.get("outcome", "NEUTRAL")
    certainty = reasoning_result.get("certainty", 0.5)
    entity_id = reasoning_result.get("entity_id", "")
    
    positive_evidence = reasoning_result.get("evidence_weights", {}).get("positive", 0)
    negative_evidence = reasoning_result.get("evidence_weights", {}).get("negative", 0)
    
    positive_conclusions = reasoning_result.get("positive_conclusions", [])
    negative_conclusions = reasoning_result.get("negative_conclusions", [])
    
    chains = reasoning_result.get("chains", [])
    
    # Start with the main recommendation
    explanation_lines = []
    
    certainty_percent = int(certainty * 100)
    if outcome == "POSITIVE":
        explanation_lines.append(f"Based on the analysis, a BUY recommendation is provided for {entity_id} with {certainty_percent}% confidence.")
    elif outcome == "NEGATIVE":
        explanation_lines.append(f"Based on the analysis, a SELL recommendation is provided for {entity_id} with {certainty_percent}% confidence.")
    else:
        explanation_lines.append(f"Based on the analysis, a HOLD recommendation is provided for {entity_id} with {certainty_percent}% confidence.")
    
    # Add information about the evidence
    explanation_lines.append("")
    explanation_lines.append(f"This recommendation is based on {len(positive_conclusions)} positive and {len(negative_conclusions)} negative factors.")
    
    # Add the key positive factors
    if positive_conclusions:
        explanation_lines.append("\nKey positive factors:")
        sorted_positive = sorted(positive_conclusions, key=lambda c: c.get("attributes", {}).get("certainty", 0), reverse=True)
        for i, conclusion in enumerate(sorted_positive[:3]):  # Top 3 positive factors
            cert = conclusion.get("attributes", {}).get("certainty", 0)
            content = conclusion.get("acep", {}).get("content", {})
            concept = content.get("concept", "")
            state = content.get("state", "")
            explanation_lines.append(f"  {i+1}. {concept} is {state} ({int(cert * 100)}% certainty)")
    
    # Add the key negative factors
    if negative_conclusions:
        explanation_lines.append("\nKey negative factors:")
        sorted_negative = sorted(negative_conclusions, key=lambda c: c.get("attributes", {}).get("certainty", 0), reverse=True)
        for i, conclusion in enumerate(sorted_negative[:3]):  # Top 3 negative factors
            cert = conclusion.get("attributes", {}).get("certainty", 0)
            content = conclusion.get("acep", {}).get("content", {})
            concept = content.get("concept", "")
            state = content.get("state", "")
            explanation_lines.append(f"  {i+1}. {concept} is {state} ({int(cert * 100)}% certainty)")
    
    # Add information about reasoning chains if available
    if chains:
        explanation_lines.append("\nReasoning process:")
        # Sort chains by certainty (highest first)
        sorted_chains = sorted(chains, key=lambda c: c.get("final_certainty", 0), reverse=True)
        
        # Describe the strongest chain
        strongest_chain = sorted_chains[0]
        steps = strongest_chain.get("steps", [])
        
        explanation_lines.append(f"  The strongest reasoning chain had {len(steps)} steps:")
        
        for i, step in enumerate(steps):
            acep = step.get("acep", {})
            content = acep.get("content", {})
            concept = content.get("concept", "")
            state = content.get("state", "")
            cert = step.get("certainty", 0)
            
            if i == len(steps) - 1:  # Final step
                explanation_lines.append(f"  - Final conclusion: {concept} is {state} ({int(cert * 100)}% certainty)")
            else:
                explanation_lines.append(f"  - Step {i+1}: {concept} is {state} ({int(cert * 100)}% certainty)")
    
    # Combine all lines into a final explanation
    return "\n".join(explanation_lines)