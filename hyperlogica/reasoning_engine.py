"""
Reasoning Engine for the Hyperlogica System

This module implements a functional-style reasoning engine that applies various
logical patterns to concepts and their relationships using hyperdimensional computing
principles for vector operations and maintains careful tracking of certainty propagation.

The reasoning engine provides core functionality for:
1. Applying logical patterns (modus ponens, modus tollens, etc.)
2. Building reasoning chains
3. Managing certainty calculations
4. Generating explanations of reasoning steps
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import logging
import uuid
from .vector_operations import (
    bind_vectors, unbind_vectors, bundle_vectors, permute_vector,
    calculate_similarity, normalize_vector, generate_vector
)

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
Vector = np.ndarray
Concept = Dict[str, Any]
Rule = Dict[str, Any]
Fact = Dict[str, Any]
State = Dict[str, Any]
VectorStore = Dict[str, Any]

# Similarity threshold for vector matching
DEFAULT_SIMILARITY_THRESHOLD = 0.7

def is_conditional(rule: Rule) -> bool:
    """
    Check if a rule is conditional (has an antecedent and consequent).
    
    Args:
        rule (Dict): Rule representation to check
        
    Returns:
        bool: True if the rule is conditional, False otherwise
    """
    return (
        rule.get("attributes", {}).get("conditional", False) or
        "antecedent" in rule.get("attributes", {}) or
        "_if_" in rule.get("identifier", "")
    )

def extract_antecedent(rule: Rule) -> str:
    """
    Extract the antecedent (condition part) from a conditional rule.
    
    Args:
        rule (Dict): Rule representation from which to extract the antecedent
        
    Returns:
        str: The antecedent text or identifier
        
    Raises:
        ValueError: If the rule is not conditional or has no identifiable antecedent
    """
    if not is_conditional(rule):
        raise ValueError("Cannot extract antecedent from a non-conditional rule")
    
    attributes = rule.get("attributes", {})
    
    # Try to get from attributes first
    if "antecedent" in attributes:
        return attributes["antecedent"]
    
    # Try to extract from identifier
    identifier = rule.get("identifier", "")
    if "_if_" in identifier:
        parts = identifier.split("_if_")
        if len(parts) >= 2:
            return parts[1]
    
    # Try to extract from rule text
    rule_text = attributes.get("rule_text", "")
    if "if" in rule_text.lower() and "then" in rule_text.lower():
        parts = rule_text.lower().split("then")
        antecedent = parts[0]
        if antecedent.startswith("if "):
            antecedent = antecedent[3:]
        return antecedent.strip()
    
    raise ValueError(f"Could not extract antecedent from rule: {rule}")

def extract_consequent(rule: Rule) -> str:
    """
    Extract the consequent (result part) from a conditional rule.
    
    Args:
        rule (Dict): Rule representation from which to extract the consequent
        
    Returns:
        str: The consequent text or identifier
        
    Raises:
        ValueError: If the rule is not conditional or has no identifiable consequent
    """
    if not is_conditional(rule):
        raise ValueError("Cannot extract consequent from a non-conditional rule")
    
    attributes = rule.get("attributes", {})
    
    # Try to get from attributes first
    if "consequent" in attributes:
        return attributes["consequent"]
    
    # Try to extract from identifier
    identifier = rule.get("identifier", "")
    if "_if_" in identifier:
        parts = identifier.split("_if_")
        if len(parts) >= 1:
            return parts[0]
    
    # Try to extract from rule text
    rule_text = attributes.get("rule_text", "")
    if "if" in rule_text.lower() and "then" in rule_text.lower():
        parts = rule_text.lower().split("then")
        if len(parts) >= 2:
            return parts[1].strip().rstrip(".")
    
    raise ValueError(f"Could not extract consequent from rule: {rule}")

def vector_matches(fact_vector: Vector, antecedent_vector: Vector, 
                  threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> bool:
    """
    Check if a fact vector matches an antecedent vector based on similarity.
    
    Args:
        fact_vector (np.ndarray): Vector representation of the fact
        antecedent_vector (np.ndarray): Vector representation of the antecedent
        threshold (float, optional): Similarity threshold for considering a match.
                                    Defaults to DEFAULT_SIMILARITY_THRESHOLD.
        
    Returns:
        bool: True if vectors are sufficiently similar, False otherwise
    """
    similarity = calculate_similarity(fact_vector, antecedent_vector)
    return similarity >= threshold

def detect_vector_type(vector: Vector) -> str:
    """
    Detect the type of a vector based on its values.
    
    Args:
        vector (np.ndarray): Vector to analyze
        
    Returns:
        str: Detected vector type ("binary", "bipolar", or "continuous")
    """
    # Check if binary (0s and 1s)
    if np.all(np.logical_or(vector == 0, vector == 1)):
        return "binary"
    # Check if bipolar (-1s and 1s)
    elif np.all(np.logical_or(vector == -1, vector == 1)):
        return "bipolar"
    # Default to continuous
    else:
        return "continuous"

def matches(fact: Fact, antecedent: str, store: VectorStore) -> Tuple[bool, float]:
    """
    Check if a fact matches an antecedent condition using vector similarity.
    
    Args:
        fact (Dict): Fact representation to check against the antecedent
        antecedent (str): The antecedent text or identifier to match
        store (Dict): Vector store for similarity comparisons
        
    Returns:
        Tuple[bool, float]: A tuple containing (match_result, similarity_score)
    """
    # First try vector-based matching if vectors are available
    if "vector" in fact and store is not None:
        fact_vector = fact.get("vector")
        
        # Find antecedent vector in store
        antecedent_vector = None
        antecedent_id = None
        
        # First try direct identifier match
        for concept_id, concept_data in store.get("concepts", {}).items():
            # Check if antecedent is directly in the identifier
            if antecedent.lower() in concept_id.lower():
                antecedent_id = concept_id
                antecedent_vector = concept_data.get("vector")
                break
        
        # If no direct match, try text-based search in metadata
        if antecedent_vector is None:
            for concept_id, concept_data in store.get("concepts", {}).items():
                concept_text = concept_data.get("metadata", {}).get("text", "").lower()
                if antecedent.lower() in concept_text:
                    antecedent_id = concept_id
                    antecedent_vector = concept_data.get("vector")
                    break
        
        # If we found an antecedent vector, check similarity
        if antecedent_vector is not None:
            similarity = calculate_similarity(fact_vector, antecedent_vector)
            match_result = similarity >= DEFAULT_SIMILARITY_THRESHOLD
            logger.debug(f"Vector similarity between fact and antecedent: {similarity:.4f} (threshold: {DEFAULT_SIMILARITY_THRESHOLD})")
            return match_result, similarity
    
    # Fallback to text-based matching
    fact_text = fact.get("attributes", {}).get("fact_text", "").lower()
    fact_id = fact.get("identifier", "").lower()
    
    # Check if antecedent text appears in fact text or identifier
    if antecedent.lower() in fact_text or antecedent.lower() in fact_id:
        logger.debug(f"Text match between fact and antecedent")
        return True, 1.0
    
    # Check specific attributes if available (domain-specific matching)
    # This is a fallback when vector matching fails
    fact_attributes = fact.get("attributes", {})
    assessment = fact_attributes.get("assessment", "").lower()
    metric_type = fact_attributes.get("metric_type", "").lower()
    
    # Simple attribute-based matching patterns
    patterns = [
        # PE ratio conditions
        (lambda: "pe ratio" in antecedent.lower() and "pe_ratio" in metric_type,
         lambda: ("low" in antecedent.lower() and "low" in assessment) or 
                 ("high" in antecedent.lower() and "high" in assessment)),
        
        # Revenue growth conditions
        (lambda: "revenue growth" in antecedent.lower() and "revenue_growth" in metric_type,
         lambda: ("high" in antecedent.lower() and "high" in assessment) or
                 ("low" in antecedent.lower() and "low" in assessment) or
                 ("negative" in antecedent.lower() and "negative" in assessment)),
        
        # Profit margin conditions
        (lambda: "profit margin" in antecedent.lower() and "profit_margin" in metric_type,
         lambda: ("high" in antecedent.lower() and "high" in assessment) or
                 ("low" in antecedent.lower() and "low" in assessment)),
    ]
    
    for condition_check, match_check in patterns:
        if condition_check() and match_check():
            logger.debug(f"Attribute match between {metric_type}:{assessment} and {antecedent.lower()}")
            return True, 0.9  # High but not perfect confidence for attribute matching
    
    return False, 0.0

def apply_modus_ponens_vector(rule_vector: Vector, fact_vector: Vector, 
                            vector_dim: int = 10000) -> Vector:
    """
    Apply modus ponens using vector operations: If P→Q and P, then Q.
    
    Args:
        rule_vector (np.ndarray): Vector representation of the conditional rule (P→Q)
        fact_vector (np.ndarray): Vector representation of the fact (P)
        vector_dim (int, optional): Dimensionality for generating vectors if needed.
                                   Defaults to 10000.
        
    Returns:
        np.ndarray: Vector representing the conclusion (Q)
    """
    # In vector symbolic architectures, modus ponens can be approximated by:
    # Q ≈ (P→Q) ⊛ P where ⊛ is circular convolution (or appropriate operation)
    # This is a simplification, but provides a reasonable approximation
    
    vector_type = detect_vector_type(rule_vector)
    binding_method = "xor" if vector_type == "binary" else "convolution"
    
    # Extract Q from the rule by unbinding P
    conclusion_vector = unbind_vectors(rule_vector, fact_vector, binding_method)
    
    # For continuous vectors, normalization is important
    if vector_type == "continuous":
        conclusion_vector = normalize_vector(conclusion_vector)
        
    return conclusion_vector

def apply_modus_ponens(rule: Rule, fact: Fact, store: VectorStore) -> Concept:
    """
    Apply modus ponens: If P→Q and P, then Q.
    Implementation using vector operations when possible, with fallback to text matching.
    
    Args:
        rule (Dict): Conditional rule representation (P→Q)
        fact (Dict): Fact representation matching the antecedent (P)
        store (Dict): Vector store for retrieving related vectors
        
    Returns:
        Dict: Derived conclusion representation (Q) with certainty
        
    Raises:
        ValueError: If the rule is not conditional or the fact doesn't match the antecedent
    """
    logger.info(f"Applying modus ponens with rule '{rule.get('identifier')}' to fact '{fact.get('identifier')}'")
    
    if not is_conditional(rule):
        raise ValueError("Rule must be conditional for modus ponens")
    
    # Check if the fact matches the rule's antecedent
    antecedent = extract_antecedent(rule)
    match_result, similarity = matches(fact, antecedent, store)
    
    if not match_result:
        raise ValueError(f"Fact '{fact.get('identifier')}' doesn't match antecedent '{antecedent}'")
    
    # Extract consequent
    try:
        consequent_text = extract_consequent(rule)
        logger.info(f"Extracted consequent: '{consequent_text}'")
    except ValueError as e:
        logger.error(f"Failed to extract consequent: {str(e)}")
        # Create a fallback consequent ID
        consequent_text = f"derived_from_{rule.get('identifier')}_and_{fact.get('identifier')}"
    
    # Create the consequent identifier
    consequent_id = create_identifier_from_text(consequent_text)
    
    # Try to use vector operations if vectors are available
    rule_vector = rule.get("vector")
    fact_vector = fact.get("vector")
    vector_dim = 10000  # Default dimension
    
    if rule_vector is not None and fact_vector is not None:
        try:
            # Get vector dimension from existing vectors
            vector_dim = rule_vector.shape[0]
            
            # Apply modus ponens using vector operations
            conclusion_vector = apply_modus_ponens_vector(rule_vector, fact_vector, vector_dim)
            logger.info(f"Generated conclusion vector using vector operations")
        except Exception as e:
            logger.warning(f"Vector operation failed: {str(e)}. Falling back to generation.")
            # Fallback: generate a new vector for the conclusion
            conclusion_vector = generate_vector(consequent_id, vector_dim)
    else:
        # If vectors aren't available, generate a vector for the conclusion
        logger.info(f"Vectors not available, generating vector for conclusion")
        
        # Try to get dimension from store configuration
        if store and "dimension" in store:
            vector_dim = store["dimension"]
        
        conclusion_vector = generate_vector(consequent_id, vector_dim)
    
    # Calculate certainty: min(certainty(P→Q), certainty(P)) × similarity
    rule_certainty = rule.get("attributes", {}).get("certainty", 1.0)
    fact_certainty = fact.get("attributes", {}).get("certainty", 1.0)
    match_certainty = similarity  # Incorporate similarity as a certainty factor
    certainty = min(rule_certainty, fact_certainty) * match_certainty
    logger.info(f"Calculated certainty: min({rule_certainty}, {fact_certainty}) × {match_certainty} = {certainty}")
    
    # Create consequent concept
    consequent = {
        "identifier": consequent_id,
        "vector": conclusion_vector,
        "type": "concept",
        "attributes": {
            "derived_from": [rule.get("identifier"), fact.get("identifier")],
            "derivation_method": "modus_ponens",
            "rule_text": rule.get("attributes", {}).get("rule_text", ""),
            "fact_text": fact.get("attributes", {}).get("fact_text", ""),
            "entity_id": fact.get("attributes", {}).get("entity_id", ""),
            "source": "derived",
            "derivation_pattern": "modus_ponens",
            "certainty": certainty,
            "text": consequent_text
        }
    }
    
    logger.info(f"Created conclusion: {consequent_id} with certainty {certainty}")
    return consequent

def apply_conjunction_introduction_vector(fact_a_vector: Vector, fact_b_vector: Vector) -> Vector:
    """
    Apply conjunction introduction using vector operations: P, Q, therefore P∧Q.
    
    Args:
        fact_a_vector (np.ndarray): Vector representation of the first fact (P)
        fact_b_vector (np.ndarray): Vector representation of the second fact (Q)
        
    Returns:
        np.ndarray: Vector representing the conjunction (P∧Q)
    """
    # In vector symbolic architectures, conjunction can be represented by bundling
    # the individual vectors (possibly with binding to role vectors, but we'll keep it simple)
    
    # Bundle the vectors for P and Q
    conjunction_vector = bundle_vectors([fact_a_vector, fact_b_vector])
    
    return conjunction_vector

def apply_conjunction_introduction(fact_a: Fact, fact_b: Fact, store: VectorStore) -> Concept:
    """
    Apply conjunction introduction: P, Q, therefore P∧Q.
    
    Args:
        fact_a (Dict): First fact representation (P)
        fact_b (Dict): Second fact representation (Q)
        store (Dict): Vector store for retrieving related vectors
        
    Returns:
        Dict: Derived conjunction representation (P∧Q) with certainty
        
    Raises:
        ValueError: If either fact is invalid or incompatible for conjunction
    """
    logger.info(f"Applying conjunction introduction with facts '{fact_a.get('identifier')}' and '{fact_b.get('identifier')}'")
    
    # Create conjunction identifier
    conjunction_id = f"{fact_a.get('identifier')}_and_{fact_b.get('identifier')}"
    
    # Use vector operations if vectors are available
    fact_a_vector = fact_a.get("vector")
    fact_b_vector = fact_b.get("vector")
    
    if fact_a_vector is not None and fact_b_vector is not None:
        try:
            # Check vector dimensions match
            if fact_a_vector.shape != fact_b_vector.shape:
                raise ValueError("Fact vectors must have the same dimensions for conjunction")
            
            # Apply conjunction using vector operations
            conjunction_vector = apply_conjunction_introduction_vector(fact_a_vector, fact_b_vector)
            logger.info(f"Generated conjunction vector using vector operations")
        except Exception as e:
            logger.warning(f"Vector operation failed: {str(e)}. Falling back to generation.")
            # Fallback: generate a new vector for the conjunction
            vector_dim = fact_a_vector.shape[0]
            conjunction_vector = generate_vector(conjunction_id, vector_dim)
    else:
        # If vectors aren't available, generate a vector for the conjunction
        logger.info(f"Vectors not available, generating vector for conjunction")
        vector_dim = 10000  # Default dimension
        
        # Try to get dimension from store configuration
        if store and "dimension" in store:
            vector_dim = store["dimension"]
            
        conjunction_vector = generate_vector(conjunction_id, vector_dim)
    
    # Calculate certainty: min(certainty(P), certainty(Q))
    fact_a_certainty = fact_a.get("attributes", {}).get("certainty", 1.0)
    fact_b_certainty = fact_b.get("attributes", {}).get("certainty", 1.0)
    certainty = min(fact_a_certainty, fact_b_certainty)
    logger.info(f"Calculated certainty: min({fact_a_certainty}, {fact_b_certainty}) = {certainty}")
    
    # Create conjunction concept
    conjunction = {
        "identifier": conjunction_id,
        "vector": conjunction_vector,
        "type": "concept",
        "attributes": {
            "components": [fact_a.get("identifier"), fact_b.get("identifier")],
            "entity_id": fact_a.get("attributes", {}).get("entity_id", 
                        fact_b.get("attributes", {}).get("entity_id", "")),
            "source": "derived",
            "derivation_pattern": "conjunction_introduction",
            "fact_a_text": fact_a.get("attributes", {}).get("fact_text", ""),
            "fact_b_text": fact_b.get("attributes", {}).get("fact_text", ""),
            "certainty": certainty,
            "text": f"{fact_a.get('attributes', {}).get('fact_text', '')} and {fact_b.get('attributes', {}).get('fact_text', '')}"
        }
    }
    
    logger.info(f"Created conjunction: {conjunction_id} with certainty {certainty}")
    return conjunction

def create_identifier_from_text(text: str) -> str:
    """
    Create a normalized identifier from text.
    
    Args:
        text (str): Text to convert to an identifier
        
    Returns:
        str: Normalized identifier with spaces replaced by underscores,
             special characters removed, and truncated if too long
    """
    # Replace non-alphanumeric chars (except spaces) with empty string
    identifier = "".join(c for c in text if c.isalnum() or c.isspace())
    
    # Convert to lowercase and replace spaces with underscores
    identifier = identifier.lower().replace(" ", "_")
    
    # Truncate if too long
    max_length = 50
    if len(identifier) > max_length:
        identifier = identifier[:max_length]
    
    return identifier

def calculate_certainty(evidence_certainties: List[float], method: str = "min") -> float:
    """
    Calculate overall certainty from multiple pieces of evidence.
    
    Args:
        evidence_certainties (List[float]): List of certainty values from different evidence sources
        method (str, optional): Method to use for combining certainties. Options include:
                              "min" (conservative), "product" (independent probabilities),
                              "noisy_or" (redundant evidence), or "weighted" (weighted average).
                              Defaults to "min".
        
    Returns:
        float: Combined certainty value between 0 and 1
        
    Raises:
        ValueError: If an invalid combination method is specified or if evidence_certainties is empty
    """
    if not evidence_certainties:
        raise ValueError("Cannot calculate certainty from empty evidence list")
    
    if method == "min":
        # Conservative approach: take the minimum certainty
        return min(evidence_certainties)
    
    elif method == "product":
        # Independent probabilities: multiply certainties
        result = 1.0
        for cert in evidence_certainties:
            result *= cert
        return result
    
    elif method == "noisy_or":
        # Noisy-OR model: 1 - product of (1 - cert)
        result = 1.0
        for cert in evidence_certainties:
            result *= (1.0 - cert)
        return 1.0 - result
    
    elif method == "weighted":
        # Default to equal weights if not provided
        weights = [1.0 / len(evidence_certainties)] * len(evidence_certainties)
        
        # Check if we need to normalize weights
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-10:
            weights = [w / weight_sum for w in weights]
        
        # Calculate weighted average
        return sum(w * c for w, c in zip(weights, evidence_certainties))
    
    else:
        raise ValueError(f"Invalid certainty combination method: {method}")

def recalibrate_certainty(raw_certainty: float, context: Dict[str, Any], method: str = "linear") -> float:
    """
    Adjust certainty based on context and method.
    
    Args:
        raw_certainty (float): Initial certainty value between 0 and 1
        context (Dict[str, Any]): Contextual information that might influence calibration,
                                such as domain knowledge or historical accuracy
        method (str, optional): Recalibration method to use. Options include:
                              "linear" (simple scaling), "sigmoid" (compress extremes),
                              "expert" (use context-specific calibration rules),
                              or "historical" (based on past accuracy).
                              Defaults to "linear".
        
    Returns:
        float: Recalibrated certainty value between 0 and 1
        
    Raises:
        ValueError: If raw_certainty is outside [0,1] or an invalid method is specified
    """
    if raw_certainty < 0 or raw_certainty > 1:
        raise ValueError(f"Raw certainty must be between 0 and 1, got {raw_certainty}")
    
    if method == "linear":
        # Simple linear scaling with optional scaling factor
        scale_factor = context.get("scale_factor", 1.0)
        bias = context.get("bias", 0.0)
        
        # Apply scale and bias, then clamp to [0,1]
        result = raw_certainty * scale_factor + bias
        return max(0.0, min(1.0, result))
    
    elif method == "sigmoid":
        # Sigmoid function to compress extremes
        # f(x) = 1 / (1 + e^(-k * (x - x0)))
        steepness = context.get("steepness", 10.0)  # Controls how sharp the sigmoid is
        midpoint = context.get("midpoint", 0.5)  # Value where sigmoid equals 0.5
        
        import math
        result = 1.0 / (1.0 + math.exp(-steepness * (raw_certainty - midpoint)))
        return result
    
    elif method == "expert":
        # Use expert-defined rules from context
        rules = context.get("calibration_rules", [])
        
        for rule in rules:
            min_val = rule.get("min", 0.0)
            max_val = rule.get("max", 1.0)
            
            if min_val <= raw_certainty <= max_val:
                adjustment = rule.get("adjustment", 0.0)
                return max(0.0, min(1.0, raw_certainty + adjustment))
        
        # If no rule matches, return the original certainty
        return raw_certainty
    
    elif method == "historical":
        # Use historical accuracy data to calibrate
        historical_data = context.get("historical_data", {})
        
        # Find the closest bin
        bins = sorted(historical_data.keys())
        if not bins:
            return raw_certainty
        
        closest_bin = min(bins, key=lambda x: abs(float(x) - raw_certainty))
        calibrated_value = historical_data.get(closest_bin, raw_certainty)
        
        return calibrated_value
    
    else:
        raise ValueError(f"Invalid certainty recalibration method: {method}")

def create_reasoning_chain(premises: List[Concept], pattern_sequence: List[Dict], store: VectorStore) -> Dict:
    """
    Create a chain of reasoning using the specified patterns.
    
    Args:
        premises (list): List of initial premise representations to start reasoning from
        pattern_sequence (list): List of reasoning patterns to apply in sequence, where each
                                pattern is a dict specifying the pattern type and parameters
        store (dict): Vector store for retrieving related vectors
        
    Returns:
        dict: Final conclusion with full reasoning trace, including intermediate
              conclusions and certainty values for each step
        
    Raises:
        ValueError: If the premises are invalid or if any reasoning step fails
    """
    if not premises:
        raise ValueError("At least one premise is required for reasoning")
    
    intermediate_results = premises.copy()
    reasoning_steps = []
    
    # Apply each reasoning pattern in sequence
    for step_idx, pattern_info in enumerate(pattern_sequence):
        pattern_type = pattern_info.get("pattern")
        if not pattern_type:
            raise ValueError(f"Missing pattern type in step {step_idx}")
        
        # Determine which function to call based on pattern type
        if pattern_type == "modus_ponens":
            if len(intermediate_results) < 2:
                raise ValueError(f"Modus ponens requires at least two premises at step {step_idx}")
            
            # Find the rule and the matching fact
            rule_idx = pattern_info.get("rule_idx", 0)
            fact_idx = pattern_info.get("fact_idx", 1)
            
            if rule_idx >= len(intermediate_results) or fact_idx >= len(intermediate_results):
                raise ValueError(f"Invalid indices for modus ponens at step {step_idx}")
            
            rule = intermediate_results[rule_idx]
            fact = intermediate_results[fact_idx]
            
            # Apply modus ponens
            try:
                conclusion = apply_modus_ponens(rule, fact, store)
                intermediate_results.append(conclusion)
                
                # Record the reasoning step
                reasoning_steps.append({
                    "step_id": step_idx,
                    "pattern": "modus_ponens",
                    "premises": [rule.get("identifier"), fact.get("identifier")],
                    "conclusion": conclusion.get("identifier"),
                    "certainty": conclusion.get("attributes", {}).get("certainty", 0.0)
                })
            except ValueError as e:
                logger.error(f"Modus ponens failed at step {step_idx}: {str(e)}")
                raise ValueError(f"Reasoning chain failed at step {step_idx}: {str(e)}") from e
                
        elif pattern_type == "conjunction_introduction":
            if len(intermediate_results) < 2:
                raise ValueError(f"Conjunction introduction requires at least two premises at step {step_idx}")
            
            # Find the facts to combine
            fact_a_idx = pattern_info.get("fact_a_idx", 0)
            fact_b_idx = pattern_info.get("fact_b_idx", 1)
            
            if fact_a_idx >= len(intermediate_results) or fact_b_idx >= len(intermediate_results):
                raise ValueError(f"Invalid indices for conjunction introduction at step {step_idx}")
            
            fact_a = intermediate_results[fact_a_idx]
            fact_b = intermediate_results[fact_b_idx]
            
            # Apply conjunction introduction
            try:
                conclusion = apply_conjunction_introduction(fact_a, fact_b, store)
                intermediate_results.append(conclusion)
                
                # Record the reasoning step
                reasoning_steps.append({
                    "step_id": step_idx,
                    "pattern": "conjunction_introduction",
                    "premises": [fact_a.get("identifier"), fact_b.get("identifier")],
                    "conclusion": conclusion.get("identifier"),
                    "certainty": conclusion.get("attributes", {}).get("certainty", 0.0)
                })
            except ValueError as e:
                logger.error(f"Conjunction introduction failed at step {step_idx}: {str(e)}")
                raise ValueError(f"Reasoning chain failed at step {step_idx}: {str(e)}") from e
        else:
            raise ValueError(f"Unknown reasoning pattern '{pattern_type}' at step {step_idx}")
    
    # Create the final reasoning result with trace
    if not intermediate_results:
        raise ValueError("Reasoning chain produced no results")
    
    final_conclusion = intermediate_results[-1]
    
    reasoning_result = {
        "conclusion": final_conclusion,
        "certainty": final_conclusion.get("attributes", {}).get("certainty", 0.0),
        "trace": {
            "premises": [p.get("identifier") for p in premises],
            "steps": reasoning_steps,
            "intermediate_results": [r.get("identifier") for r in intermediate_results]
        }
    }
    
    return reasoning_result

def generate_explanation(conclusion: Concept, reasoning_steps: List[Dict], store: VectorStore) -> str:
    """
    Generate a natural language explanation of the reasoning process.
    
    Args:
        conclusion (Dict): The final conclusion concept
        reasoning_steps (List[Dict]): List of reasoning steps that led to the conclusion
        store (Dict): Vector store for retrieving related concepts
        
    Returns:
        str: Natural language explanation of the reasoning process
    """
    if not reasoning_steps:
        # This is a base concept without derived reasoning
        concept_text = conclusion.get("attributes", {}).get("text", conclusion.get("identifier", ""))
        certainty = conclusion.get("attributes", {}).get("certainty", 1.0)
        certainty_text = f"{certainty:.1%}" if certainty < 1.0 else "high"
        
        return f"The system determined: {concept_text}, with {certainty_text} certainty."
    
    # Build explanation from reasoning steps
    explanation = [f"Based on the available information, the system reached the following conclusion:"]
    explanation.append("")
    
    # Add the final conclusion first
    concept_text = conclusion.get("attributes", {}).get("text", conclusion.get("identifier", ""))
    certainty = conclusion.get("attributes", {}).get("certainty", 1.0)
    explanation.append(f"CONCLUSION: {concept_text} (Certainty: {certainty:.1%})")
    explanation.append("")
    explanation.append("This was determined through the following reasoning process:")
    
    # Add each reasoning step
    for step_idx, step in enumerate(reasoning_steps):
        pattern = step.get("pattern", "unknown")
        conclusion_id = step.get("conclusion", "")
        premise_ids = step.get("premises", [])
        step_certainty = step.get("certainty", 0.0)
        
        # Get the text representations of premises and conclusion
        premise_texts = []
        for premise_id in premise_ids:
            # Try to find the concept in the store
            concept = None
            if store is not None:
                for concept_id, concept_data in store.get("concepts", {}).items():
                    if concept_id == premise_id:
                        concept = concept_data
                        break
            
            if concept:
                text = concept.get("metadata", {}).get("text", premise_id)
            else:
                text = premise_id
            premise_texts.append(text)
        
        # Get the conclusion text
        conclusion_text = conclusion_id
        if store is not None:
            for concept_id, concept_data in store.get("concepts", {}).items():
                if concept_id == conclusion_id:
                    conclusion_text = concept_data.get("metadata", {}).get("text", conclusion_id)
                    break
        
        # Format the step explanation based on the pattern
        if pattern == "modus_ponens":
            explanation.append(f"Step {step_idx+1}: Applied modus ponens (If P then Q, P is true, therefore Q is true)")
            explanation.append(f"  * Rule: {premise_texts[0]}")
            explanation.append(f"  * Fact: {premise_texts[1]}")
            explanation.append(f"  * Therefore: {conclusion_text} (Certainty: {step_certainty:.1%})")
        elif pattern == "conjunction_introduction":
            explanation.append(f"Step {step_idx+1}: Applied conjunction (P is true, Q is true, therefore P and Q are true)")
            explanation.append(f"  * Fact 1: {premise_texts[0]}")
            explanation.append(f"  * Fact 2: {premise_texts[1]}")
            explanation.append(f"  * Therefore: {conclusion_text} (Certainty: {step_certainty:.1%})")
        else:
            explanation.append(f"Step {step_idx+1}: Applied {pattern}")
            explanation.append(f"  * Premises: {', '.join(premise_texts)}")
            explanation.append(f"  * Conclusion: {conclusion_text} (Certainty: {step_certainty:.1%})")
        
        explanation.append("")
    
    return "\n".join(explanation)

def record_reasoning_step(
    pattern: str, 
    premises: List[str], 
    conclusion: str, 
    certainty: float, 
    step_id: Optional[int] = None
) -> Dict:
    """
    Record a reasoning step for explanation purposes.
    
    Args:
        pattern (str): Reasoning pattern used (e.g., "modus_ponens", "conjunction")
        premises (List[str]): List of premise identifiers used in the reasoning step
        conclusion (str): Derived conclusion identifier
        certainty (float): Calculated certainty for the conclusion
        step_id (int, optional): Optional step identifier. If None, a UUID will be generated.
                               Defaults to None.
        
    Returns:
        Dict: Dictionary containing the reasoning step information
    """
    if step_id is None:
        # Generate a unique step ID if none provided
        step_id = str(uuid.uuid4())
    
    step = {
        "step_id": step_id,
        "pattern": pattern,
        "premises": premises,
        "conclusion": conclusion,
        "certainty": certainty,
        "timestamp": str(uuid.uuid1())  # Include a timestamp for ordering
    }
    
    return step