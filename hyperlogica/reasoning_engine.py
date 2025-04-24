"""
Reasoning Engine for the Hyperlogica System

This module implements a functional-style reasoning engine that applies various
logical patterns to concepts and their relationships. It uses hyperdimensional
computing principles for vector operations and maintains careful tracking of
certainty propagation.

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

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
Vector = np.ndarray
Concept = Dict[str, Any]
Rule = Dict[str, Any]
Fact = Dict[str, Any]
State = Dict[str, Any]
VectorStore = Dict[str, Any]

def is_conditional(rule: Rule) -> bool:
    """
    Check if a rule is conditional (has an antecedent and consequent).
    
    Args:
        rule (Dict): Rule representation to check
        
    Returns:
        bool: True if the rule is conditional, False otherwise
    """
    return (
        rule.get("metadata", {}).get("conditional", False) or
        "antecedent" in rule.get("metadata", {}) or
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
    
    metadata = rule.get("metadata", {})
    
    # Try to get from metadata first
    if "antecedent" in metadata:
        return metadata["antecedent"]
    
    # Try to extract from identifier
    identifier = rule.get("identifier", "")
    if "_if_" in identifier:
        parts = identifier.split("_if_")
        if len(parts) >= 2:
            return parts[1]
    
    # Try to extract from rule text
    rule_text = metadata.get("rule_text", "")
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
    
    metadata = rule.get("metadata", {})
    
    # Try to get from metadata first
    if "consequent" in metadata:
        return metadata["consequent"]
    
    # Try to extract from identifier
    identifier = rule.get("identifier", "")
    if "_if_" in identifier:
        parts = identifier.split("_if_")
        if len(parts) >= 1:
            return parts[0]
    
    # Try to extract from rule text
    rule_text = metadata.get("rule_text", "")
    if "if" in rule_text.lower() and "then" in rule_text.lower():
        parts = rule_text.lower().split("then")
        if len(parts) >= 2:
            return parts[1].strip().rstrip(".")
    
    raise ValueError(f"Could not extract consequent from rule: {rule}")

def matches(fact: Fact, antecedent: str, store: VectorStore) -> bool:
    """
    Check if a fact matches an antecedent condition.
    
    Args:
        fact (Dict): Fact representation to check against the antecedent
        antecedent (str): The antecedent text or identifier to match
        store (Dict): Vector store for similarity comparisons
        
    Returns:
        bool: True if the fact matches the antecedent, False otherwise
    """
    # Direct string matching in fact text or identifier
    fact_text = fact.get("metadata", {}).get("fact_text", "").lower()
    fact_id = fact.get("identifier", "").lower()
    antecedent_lower = antecedent.lower()
    
    if antecedent_lower in fact_text or antecedent_lower in fact_id:
        logger.debug(f"Direct string match between fact and antecedent: {antecedent_lower}")
        return True
    
    # Check for semantic similarity using vector representations
    if "vector" in fact and store is not None:
        # Get antecedent vector if available
        antecedent_vector = None
        for concept_id, concept in store.get("concepts", {}).items():
            if antecedent_lower in concept_id.lower():
                antecedent_vector = concept.get("vector")
                break
        
        if antecedent_vector is not None:
            fact_vector = fact.get("vector")
            similarity = calculate_vector_similarity(fact_vector, antecedent_vector)
            threshold = 0.7  # Configurable threshold
            if similarity >= threshold:
                logger.debug(f"Vector similarity match: {similarity} >= {threshold}")
                return True
    
    # Check specific attributes if available
    fact_metadata = fact.get("metadata", {})
    assessment = fact_metadata.get("assessment", "").lower()
    metric_type = fact_metadata.get("metric_type", "").lower()
    
    # Simplified pattern matching for common conditions
    patterns = [
        # PE ratio conditions
        (lambda: "pe ratio" in antecedent_lower and "pe_ratio" in metric_type,
         lambda: ("low" in antecedent_lower and "low" in assessment) or 
                 ("high" in antecedent_lower and "high" in assessment)),
        
        # Revenue growth conditions
        (lambda: "revenue growth" in antecedent_lower and "revenue_growth" in metric_type,
         lambda: ("high" in antecedent_lower and "high" in assessment) or
                 ("low" in antecedent_lower and "low" in assessment) or
                 ("negative" in antecedent_lower and "negative" in assessment)),
        
        # Profit margin conditions
        (lambda: "profit margin" in antecedent_lower and "profit_margin" in metric_type,
         lambda: ("high" in antecedent_lower and "high" in assessment) or
                 ("low" in antecedent_lower and "low" in assessment)),
        
        # Debt ratio conditions
        (lambda: ("debt" in antecedent_lower or "leverage" in antecedent_lower) and 
                 "debt_to_equity" in metric_type,
         lambda: ("low" in antecedent_lower and "low" in assessment) or
                 ("high" in antecedent_lower and "high" in assessment)),
        
        # ROE conditions
        (lambda: "return on equity" in antecedent_lower and "return_on_equity" in metric_type,
         lambda: ("high" in antecedent_lower and "high" in assessment) or
                 ("low" in antecedent_lower and "low" in assessment)),
        
        # Price movement conditions
        (lambda: "price movement" in antecedent_lower and "price_movement" in metric_type,
         lambda: ("positive" in antecedent_lower and "positive" in assessment) or
                 ("negative" in antecedent_lower and "negative" in assessment)),
                 
        # Analyst sentiment conditions
        (lambda: "analyst" in antecedent_lower and "analyst_sentiment" in metric_type,
         lambda: ("high" in antecedent_lower and "high" in assessment) or
                 ("low" in antecedent_lower and "low" in assessment))
    ]
    
    for condition_check, match_check in patterns:
        if condition_check() and match_check():
            logger.debug(f"Pattern match between {metric_type}:{assessment} and {antecedent_lower}")
            return True
    
    return False

def calculate_vector_similarity(vec1: Vector, vec2: Vector, method: str = "cosine") -> float:
    """
    Calculate similarity between two vectors.
    
    Args:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector
        method (str, optional): Similarity calculation method. Options are 
                               "cosine" or "hamming". Defaults to "cosine".
        
    Returns:
        float: Similarity score between 0 and 1, where 1 indicates identical vectors
               and 0 indicates orthogonal vectors
        
    Raises:
        ValueError: If vectors have different dimensions or if an invalid similarity 
                   method is specified
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same dimensions")
    
    if method == "cosine":
        # Normalize vectors to avoid numerical issues
        vec1_normalized = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_normalized = vec2 / (np.linalg.norm(vec2) + 1e-10)
        # Calculate cosine similarity
        return float(np.dot(vec1_normalized, vec2_normalized))
    elif method == "hamming":
        # For binary vectors, calculate normalized Hamming similarity
        if not np.array_equal(vec1, vec1.astype(bool)) or not np.array_equal(vec2, vec2.astype(bool)):
            logger.warning("Using Hamming similarity with non-binary vectors")
        
        hamming_distance = np.sum(vec1 != vec2)
        return float(1 - hamming_distance / len(vec1))
    else:
        raise ValueError(f"Invalid similarity method: {method}")

def generate_vector_for_concept(
    concept_identifier: str, 
    vector_dimension: int = 10000,
    seed: Optional[int] = None
) -> Vector:
    """
    Generate a deterministic vector representation for a concept identifier.
    
    Args:
        concept_identifier (str): Identifier to generate vector for
        vector_dimension (int, optional): Dimension of the vector. Defaults to 10000.
        seed (int, optional): Seed for random number generation. If None, a hash
                             of the identifier will be used. Defaults to None.
        
    Returns:
        np.ndarray: A normalized vector of specified dimension
    """
    # Create a hash of the text to use as a seed if none provided
    if seed is None:
        concept_hash = hash(concept_identifier) % (2**32)
    else:
        concept_hash = seed
    
    # Set the random seed for reproducibility
    np.random.seed(concept_hash)
    
    # Generate a high-dimensional vector
    vector = np.random.normal(0, 1, vector_dimension)
    
    # Normalize to unit length
    vector = vector / (np.linalg.norm(vector) + 1e-10)
    
    return vector

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
                    "certainty": conclusion.get("certainty", 0.0)
                })
            except ValueError as e:
                logger.error(f"Modus ponens failed at step {step_idx}: {str(e)}")
                raise ValueError(f"Reasoning chain failed at step {step_idx}: {str(e)}") from e
                
        elif pattern_type == "modus_tollens":
            if len(intermediate_results) < 2:
                raise ValueError(f"Modus tollens requires at least two premises at step {step_idx}")
            
            # Find the rule and the negated fact
            rule_idx = pattern_info.get("rule_idx", 0)
            negated_fact_idx = pattern_info.get("negated_fact_idx", 1)
            
            if rule_idx >= len(intermediate_results) or negated_fact_idx >= len(intermediate_results):
                raise ValueError(f"Invalid indices for modus tollens at step {step_idx}")
            
            rule = intermediate_results[rule_idx]
            negated_fact = intermediate_results[negated_fact_idx]
            
            # Apply modus tollens
            try:
                conclusion = apply_modus_tollens(rule, negated_fact, store)
                intermediate_results.append(conclusion)
                
                # Record the reasoning step
                reasoning_steps.append({
                    "step_id": step_idx,
                    "pattern": "modus_tollens",
                    "premises": [rule.get("identifier"), negated_fact.get("identifier")],
                    "conclusion": conclusion.get("identifier"),
                    "certainty": conclusion.get("certainty", 0.0)
                })
            except ValueError as e:
                logger.error(f"Modus tollens failed at step {step_idx}: {str(e)}")
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
                    "certainty": conclusion.get("certainty", 0.0)
                })
            except ValueError as e:
                logger.error(f"Conjunction introduction failed at step {step_idx}: {str(e)}")
                raise ValueError(f"Reasoning chain failed at step {step_idx}: {str(e)}") from e
                
        elif pattern_type == "disjunctive_syllogism":
            if len(intermediate_results) < 2:
                raise ValueError(f"Disjunctive syllogism requires at least two premises at step {step_idx}")
            
            # Find the disjunction and the negated fact
            disjunction_idx = pattern_info.get("disjunction_idx", 0)
            negated_fact_idx = pattern_info.get("negated_fact_idx", 1)
            
            if disjunction_idx >= len(intermediate_results) or negated_fact_idx >= len(intermediate_results):
                raise ValueError(f"Invalid indices for disjunctive syllogism at step {step_idx}")
            
            disjunction = intermediate_results[disjunction_idx]
            negated_fact = intermediate_results[negated_fact_idx]
            
            # Apply disjunctive syllogism
            try:
                conclusion = apply_disjunctive_syllogism(disjunction, negated_fact, store)
                intermediate_results.append(conclusion)
                
                # Record the reasoning step
                reasoning_steps.append({
                    "step_id": step_idx,
                    "pattern": "disjunctive_syllogism",
                    "premises": [disjunction.get("identifier"), negated_fact.get("identifier")],
                    "conclusion": conclusion.get("identifier"),
                    "certainty": conclusion.get("certainty", 0.0)
                })
            except ValueError as e:
                logger.error(f"Disjunctive syllogism failed at step {step_idx}: {str(e)}")
                raise ValueError(f"Reasoning chain failed at step {step_idx}: {str(e)}") from e
        else:
            raise ValueError(f"Unknown reasoning pattern '{pattern_type}' at step {step_idx}")
    
    # Create the final reasoning result with trace
    if not intermediate_results:
        raise ValueError("Reasoning chain produced no results")
    
    final_conclusion = intermediate_results[-1]
    
    reasoning_result = {
        "conclusion": final_conclusion,
        "certainty": final_conclusion.get("certainty", 0.0),
        "trace": {
            "premises": [p.get("identifier") for p in premises],
            "steps": reasoning_steps,
            "intermediate_results": [r.get("identifier") for r in intermediate_results]
        }
    }
    
    return reasoning_result

def create_concept(
    identifier: str,
    vector: Optional[Vector] = None,
    metadata: Optional[Dict[str, Any]] = None,
    certainty: float = 1.0,
    vector_dimension: int = 10000
) -> Concept:
    """
    Create a new concept representation.
    
    Args:
        identifier (str): Unique identifier for the concept
        vector (np.ndarray, optional): Vector representation. If None, one will be
                                      generated from the identifier. Defaults to None.
        metadata (Dict, optional): Additional metadata for the concept. Defaults to None.
        certainty (float, optional): Certainty value between 0 and 1. Defaults to 1.0.
        vector_dimension (int, optional): Dimension for generated vector if none provided.
                                         Defaults to 10000.
        
    Returns:
        Dict: A new concept representation
        
    Raises:
        ValueError: If certainty is not between 0 and 1
    """
    if certainty < 0 or certainty > 1:
        raise ValueError("Certainty must be between 0 and 1")
    
    if metadata is None:
        metadata = {}
    
    if vector is None:
        vector = generate_vector_for_concept(identifier, vector_dimension)
    
    return {
        "identifier": identifier,
        "vector": vector,
        "metadata": metadata,
        "certainty": certainty
    }

def apply_modus_ponens(rule: Rule, fact: Fact, store: VectorStore) -> Concept:
    """
    Apply modus ponens: If P→Q and P, then Q.
    
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
    
    antecedent = extract_antecedent(rule)
    if not matches(fact, antecedent, store):
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
    
    # Get or generate vector for consequent
    consequent_vector = None
    # Try to find the concept in the store first
    if store is not None:
        for concept_id, concept in store.get("concepts", {}).items():
            if consequent_id == concept_id:
                consequent_vector = concept.get("vector")
                break
                
    if consequent_vector is None:
        # Need to generate a new vector
        vector_dimension = fact.get("vector").shape[0] if fact.get("vector") is not None else 10000
        consequent_vector = generate_vector_for_concept(consequent_id, vector_dimension)
    
    # Calculate certainty: min(certainty(P→Q), certainty(P))
    rule_certainty = rule.get("certainty", 1.0)
    fact_certainty = fact.get("certainty", 1.0)
    certainty = min(rule_certainty, fact_certainty)
    logger.info(f"Calculated certainty: min({rule_certainty}, {fact_certainty}) = {certainty}")
    
    # Create consequent concept
    consequent = create_concept(
        identifier=consequent_id,
        vector=consequent_vector,
        metadata={
            "derived_from": [rule.get("identifier"), fact.get("identifier")],
            "derivation_method": "modus_ponens",
            "rule_text": rule.get("metadata", {}).get("rule_text", ""),
            "fact_text": fact.get("metadata", {}).get("fact_text", ""),
            "ticker": fact.get("metadata", {}).get("ticker", ""),
            "source": "derived",
            "derivation_pattern": "modus_ponens"
        },
        certainty=certainty
    )
    
    logger.info(f"Created conclusion: {consequent_id} with certainty {certainty}")
    return consequent

def apply_modus_tollens(rule: Rule, negated_fact: Fact, store: VectorStore) -> Concept:
    """
    Apply modus tollens: If P→Q and ¬Q, then ¬P.
    
    Args:
        rule (Dict): Conditional rule representation (P→Q)
        negated_fact (Dict): Negated fact representation matching the consequent (¬Q)
        store (Dict): Vector store for retrieving related vectors
        
    Returns:
        Dict: Derived negated conclusion representation (¬P) with certainty
        
    Raises:
        ValueError: If the rule is not conditional or the negated fact doesn't match the consequent
    """
    logger.info(f"Applying modus tollens with rule '{rule.get('identifier')}' and negated fact '{negated_fact.get('identifier')}'")
    
    if not is_conditional(rule):
        raise ValueError("Rule must be conditional for modus tollens")
    
    # Extract consequent from rule
    try:
        consequent_text = extract_consequent(rule)
        logger.info(f"Extracted consequent: '{consequent_text}'")
    except ValueError as e:
        logger.error(f"Failed to extract consequent: {str(e)}")
        raise ValueError("Failed to extract consequent from rule") from e
    
    # Check if negated fact matches negation of consequent
    negated_fact_text = negated_fact.get("metadata", {}).get("fact_text", "").lower()
    
    # This is a simplified check - in a real system, we would need more sophisticated negation handling
    if not ("not " + consequent_text.lower() in negated_fact_text or 
            "no " + consequent_text.lower() in negated_fact_text or 
            consequent_text.lower() + " is false" in negated_fact_text):
        # Also try vector similarity if direct match fails
        matches_consequent = False
        if "vector" in negated_fact and store is not None:
            # Try to find consequent vector
            consequent_vector = None
            for concept_id, concept in store.get("concepts", {}).items():
                if consequent_text.lower() in concept_id.lower():
                    consequent_vector = concept.get("vector")
                    # Need to negate this vector
                    if consequent_vector is not None:
                        consequent_vector = -consequent_vector
                    break
                    
            if consequent_vector is not None:
                negated_fact_vector = negated_fact.get("vector")
                similarity = calculate_vector_similarity(negated_fact_vector, consequent_vector)
                threshold = 0.7  # Configurable threshold
                if similarity >= threshold:
                    logger.debug(f"Vector similarity match for negation: {similarity} >= {threshold}")
                    matches_consequent = True
        
        if not matches_consequent:
            raise ValueError(f"Negated fact doesn't match negation of consequent '{consequent_text}'")
    
    # Extract antecedent
    try:
        antecedent_text = extract_antecedent(rule)
        logger.info(f"Extracted antecedent: '{antecedent_text}'")
    except ValueError as e:
        logger.error(f"Failed to extract antecedent: {str(e)}")
        raise ValueError("Failed to extract antecedent from rule") from e
    
    # Create negated antecedent identifier
    negated_antecedent_id = create_identifier_from_text("not_" + antecedent_text)
    
    # Generate vector for negated antecedent
    vector_dimension = negated_fact.get("vector").shape[0] if negated_fact.get("vector") is not None else 10000
    negated_antecedent_vector = generate_vector_for_concept(negated_antecedent_id, vector_dimension)
    
    # Calculate certainty: min(certainty(P→Q), certainty(¬Q))
    rule_certainty = rule.get("certainty", 1.0)
    negated_fact_certainty = negated_fact.get("certainty", 1.0)
    certainty = min(rule_certainty, negated_fact_certainty)
    logger.info(f"Calculated certainty: min({rule_certainty}, {negated_fact_certainty}) = {certainty}")
    
    # Create negated antecedent concept
    negated_antecedent = create_concept(
        identifier=negated_antecedent_id,
        vector=negated_antecedent_vector,
        metadata={
            "derived_from": [rule.get("identifier"), negated_fact.get("identifier")],
            "derivation_method": "modus_tollens",
            "rule_text": rule.get("metadata", {}).get("rule_text", ""),
            "fact_text": negated_fact.get("metadata", {}).get("fact_text", ""),
            "ticker": negated_fact.get("metadata", {}).get("ticker", ""),
            "source": "derived",
            "derivation_pattern": "modus_tollens"
        },
        certainty=certainty
    )
    
    logger.info(f"Created negated conclusion: {negated_antecedent_id} with certainty {certainty}")
    return negated_antecedent

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

def explain_reasoning(concept: Concept, reasoning_steps: List[Dict], store: VectorStore) -> Dict[str, Any]:
    """
    Generate explanation data for the reasoning process leading to a concept.
    
    Args:
        concept (Dict): The concept to explain
        reasoning_steps (List[Dict]): List of reasoning steps recorded during inference
        store (Dict): Vector store containing related concepts
        
    Returns:
        Dict[str, Any]: Structured explanation of the reasoning process
    """
    if not concept:
        return {"type": "empty", "message": "No concept to explain"}
    
    # Check if this is a base concept (not derived)
    if "derived_from" not in concept.get("metadata", {}) and "components" not in concept.get("metadata", {}):
        return {
            "type": "base_concept", 
            "identifier": concept.get("identifier", ""),
            "text": concept.get("metadata", {}).get("fact_text", concept.get("metadata", {}).get("rule_text", "")),
            "certainty": concept.get("certainty", 0.0)
        }
    
    # Find relevant reasoning steps for this concept
    relevant_steps = []
    for step in reasoning_steps:
        if step.get("conclusion") == concept.get("identifier"):
            relevant_steps.append(step)
    
    if not relevant_steps:
        # No recorded reasoning steps, check if it's a conjunction
        if "components" in concept.get("metadata", {}):
            return {
                "type": "conjunction", 
                "components": concept.get("metadata", {}).get("components", []),
                "certainty": concept.get("certainty", 0.0)
            }
        
        # Check if it has derivation info but no recorded steps
        if "derived_from" in concept.get("metadata", {}):
            return {
                "type": "derived_concept",
                "identifier": concept.get("identifier", ""),
                "derived_from": concept.get("metadata", {}).get("derived_from", []),
                "derivation_method": concept.get("metadata", {}).get("derivation_method", "unknown"),
                "certainty": concept.get("certainty", 0.0)
            }
        
        # Default fallback
        return {
            "type": "concept",
            "identifier": concept.get("identifier", ""),
            "certainty": concept.get("certainty", 0.0)
        }
    
    # Build explanation from reasoning steps
    explanations = []
    for step in relevant_steps:
        pattern = step.get("pattern", "unknown")
        premises = step.get("premises", [])
        conclusion = step.get("conclusion", "")
        step_certainty = step.get("certainty", 0.0)
        
        # Get text representations if available
        premise_texts = []
        for premise_id in premises:
            if store is not None:
                premise_concept = store.get("concepts", {}).get(premise_id, {})
                premise_text = premise_concept.get("metadata", {}).get("fact_text", 
                               premise_concept.get("metadata", {}).get("rule_text", premise_id))
                premise_texts.append(premise_text)
            else:
                premise_texts.append(premise_id)
        
        explanation = {
            "pattern": pattern,
            "premises": premises,
            "premise_texts": premise_texts,
            "conclusion": conclusion,
            "conclusion_text": concept.get("metadata", {}).get("fact_text", 
                               concept.get("metadata", {}).get("rule_text", conclusion)),
            "certainty": step_certainty
        }
        
        explanations.append(explanation)
    
    return {
        "type": "reasoning_chain", 
        "steps": explanations, 
        "final_certainty": concept.get("certainty", 0.0),
        "identifier": concept.get("identifier", "")
    }

def format_explanation(explanation: Dict[str, Any]) -> str:
    """
    Format a reasoning explanation as a human-readable string.
    
    Args:
        explanation (Dict[str, Any]): The explanation data structure
        
    Returns:
        str: Human-readable explanation text
    """
    explanation_type = explanation.get("type", "unknown")
    
    if explanation_type == "empty":
        return "No explanation available."
    
    elif explanation_type == "base_concept":
        text = explanation.get("text", explanation.get("identifier", ""))
        certainty = explanation.get("certainty", 0.0)
        return f"Base fact: {text} (certainty: {certainty:.2f})"
    
    elif explanation_type == "conjunction":
        components = explanation.get("components", [])
        certainty = explanation.get("certainty", 0.0)
        return f"Conjunction of concepts: {', '.join(components)} (certainty: {certainty:.2f})"
    
    elif explanation_type == "derived_concept":
        identifier = explanation.get("identifier", "")
        derived_from = explanation.get("derived_from", [])
        method = explanation.get("derivation_method", "unknown")
        certainty = explanation.get("certainty", 0.0)
        return f"Derived concept: {identifier} from {', '.join(derived_from)} using {method} (certainty: {certainty:.2f})"
    
    elif explanation_type == "reasoning_chain":
        steps = explanation.get("steps", [])
        final_certainty = explanation.get("final_certainty", 0.0)
        
        result = ["Reasoning chain:"]
        for i, step in enumerate(steps):
            pattern = step.get("pattern", "unknown")
            premise_texts = step.get("premise_texts", [])
            conclusion_text = step.get("conclusion_text", step.get("conclusion", ""))
            step_certainty = step.get("certainty", 0.0)
            
            result.append(f"  Step {i+1}: {pattern}")
            result.append(f"    Premises: {'; '.join(premise_texts)}")
            result.append(f"    Conclusion: {conclusion_text} (certainty: {step_certainty:.2f})")
        
        result.append(f"Final certainty: {final_certainty:.2f}")
        return "\n".join(result)
    
    else:
        # Default fallback
        return f"Explanation type '{explanation_type}': {json.dumps(explanation, indent=2)}"

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
    
    # Bundle vectors for conjunction (simple average approach)
    fact_a_vector = fact_a.get("vector")
    fact_b_vector = fact_b.get("vector")
    
    if fact_a_vector is None or fact_b_vector is None:
        raise ValueError("Both facts must have vector representations for conjunction")
    
    if fact_a_vector.shape != fact_b_vector.shape:
        raise ValueError("Fact vectors must have the same dimensions for conjunction")
    
    conjunction_vector = (fact_a_vector + fact_b_vector) / 2
    # Normalize the conjunction vector
    conjunction_vector = conjunction_vector / (np.linalg.norm(conjunction_vector) + 1e-10)
    
    # Calculate certainty: min(certainty(P), certainty(Q))
    fact_a_certainty = fact_a.get("certainty", 1.0)
    fact_b_certainty = fact_b.get("certainty", 1.0)
    certainty = min(fact_a_certainty, fact_b_certainty)
    logger.info(f"Calculated certainty: min({fact_a_certainty}, {fact_b_certainty}) = {certainty}")
    
    # Create conjunction concept
    conjunction = create_concept(
        identifier=conjunction_id,
        vector=conjunction_vector,
        metadata={
            "components": [fact_a.get("identifier"), fact_b.get("identifier")],
            "ticker": fact_a.get("metadata", {}).get("ticker", fact_b.get("metadata", {}).get("ticker", "")),
            "source": "derived",
            "derivation_pattern": "conjunction_introduction",
            "fact_a_text": fact_a.get("metadata", {}).get("fact_text", ""),
            "fact_b_text": fact_b.get("metadata", {}).get("fact_text", "")
        },
        certainty=certainty
    )
    
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

def apply_disjunctive_syllogism(disjunction: Concept, negated_fact: Fact, store: VectorStore) -> Concept:
    """
    Apply disjunctive syllogism: P∨Q, ¬P, therefore Q.
    
    Args:
        disjunction (Dict): Disjunction representation (P∨Q)
        negated_fact (Dict): Negated fact representation matching one disjunct (¬P)
        store (Dict): Vector store for retrieving related vectors
        
    Returns:
        Dict: Derived conclusion representation (Q) with certainty
        
    Raises:
        ValueError: If the disjunction is invalid or the negated fact doesn't match either disjunct
    """
    logger.info(f"Applying disjunctive syllogism with disjunction '{disjunction.get('identifier')}' and negated fact '{negated_fact.get('identifier')}'")
    
    # Extract disjuncts from disjunction
    disjuncts = disjunction.get("metadata", {}).get("disjuncts", [])
    if not disjuncts or len(disjuncts) < 2:
        raise ValueError("Disjunction must contain at least two disjuncts")
    
    # Determine which disjunct is being negated
    negated_disjunct = None
    for disjunct in disjuncts:
        disjunct_text = disjunct.get("text", "").lower()
        negated_fact_text = negated_fact.get("metadata", {}).get("fact_text", "").lower()
        
        # Simplified negation check
        if ("not " + disjunct_text in negated_fact_text or 
            "no " + disjunct_text in negated_fact_text or 
            disjunct_text + " is false" in negated_fact_text):
            negated_disjunct = disjunct
            break
    
    if negated_disjunct is None:
        # Try vector similarity if direct match fails
        if "vector" in negated_fact and store is not None:
            for disjunct in disjuncts:
                disjunct_id = disjunct.get("identifier")
                disjunct_vector = None
                
                # Find the disjunct vector in the store
                for concept_id, concept in store.get("concepts", {}).items():
                    if disjunct_id == concept_id:
                        disjunct_vector = concept.get("vector")
                        # Need to negate this vector
                        if disjunct_vector is not None:
                            disjunct_vector = -disjunct_vector
                        break
                
                if disjunct_vector is not None:
                    negated_fact_vector = negated_fact.get("vector")
                    similarity = calculate_vector_similarity(negated_fact_vector, disjunct_vector)
                    threshold = 0.7  # Configurable threshold
                    if similarity >= threshold:
                        logger.debug(f"Vector similarity match for negation: {similarity} >= {threshold}")
                        negated_disjunct = disjunct
                        break
        
        if negated_disjunct is None:
            raise ValueError("Negated fact doesn't match any disjunct in the disjunction")
    
    # Find the non-negated disjuncts
    remaining_disjuncts = [d for d in disjuncts if d != negated_disjunct]
    if not remaining_disjuncts:
        raise ValueError("No remaining disjuncts after eliminating the negated one")
    
    # If there's only one remaining disjunct, it's our conclusion
    if len(remaining_disjuncts) == 1:
        conclusion_disjunct = remaining_disjuncts[0]
        conclusion_id = conclusion_disjunct.get("identifier")
        
        # Get or generate vector for conclusion
        conclusion_vector = None
        if store is not None:
            for concept_id, concept in store.get("concepts", {}).items():
                if conclusion_id == concept_id:
                    conclusion_vector = concept.get("vector")
                    break
        
        if conclusion_vector is None:
            vector_dimension = negated_fact.get("vector").shape[0] if negated_fact.get("vector") is not None else 10000
            conclusion_vector = generate_vector_for_concept(conclusion_id, vector_dimension)
        
        # Calculate certainty: min(certainty(P∨Q), certainty(¬P))
        disjunction_certainty = disjunction.get("certainty", 1.0)
        negated_fact_certainty = negated_fact.get("certainty", 1.0)
        certainty = min(disjunction_certainty, negated_fact_certainty)
        logger.info(f"Calculated certainty: min({disjunction_certainty}, {negated_fact_certainty}) = {certainty}")
        
        # Create conclusion concept
        conclusion = create_concept(
            identifier=conclusion_id,
            vector=conclusion_vector,
            metadata={
                "derived_from": [disjunction.get("identifier"), negated_fact.get("identifier")],
                "derivation_method": "disjunctive_syllogism",
                "disjunction_text": disjunction.get("metadata", {}).get("disjunction_text", ""),
                "negated_fact_text": negated_fact.get("metadata", {}).get("fact_text", ""),
                "ticker": negated_fact.get("metadata", {}).get("ticker", ""),
                "source": "derived",
                "derivation_pattern": "disjunctive_syllogism"
            },
            certainty=certainty
        )
        
        logger.info(f"Created conclusion: {conclusion_id} with certainty {certainty}")
        return conclusion
    else:
        # Multiple remaining disjuncts, create a new disjunction with them
        new_disjunction_id = "_or_".join([d.get("identifier") for d in remaining_disjuncts])
        
        # Bundle the vectors of remaining disjuncts
        remaining_vectors = []
        for disjunct in remaining_disjuncts:
            disjunct_id = disjunct.get("identifier")
            disjunct_vector = None
            
            if store is not None:
                for concept_id, concept in store.get("concepts", {}).items():
                    if disjunct_id == concept_id:
                        disjunct_vector = concept.get("vector")
                        break
            
            if disjunct_vector is not None:
                remaining_vectors.append(disjunct_vector)
        
        if not remaining_vectors:
            # Fallback: generate a new vector
            vector_dimension = negated_fact.get("vector").shape[0] if negated_fact.get("vector") is not None else 10000
            new_disjunction_vector = generate_vector_for_concept(new_disjunction_id, vector_dimension)
        else:
            # Average the remaining vectors
            new_disjunction_vector = np.mean(np.array(remaining_vectors), axis=0)
            # Normalize the vector
            new_disjunction_vector = new_disjunction_vector / (np.linalg.norm(new_disjunction_vector) + 1e-10)
        
        # Calculate certainty: min(certainty(P∨Q), certainty(¬P))
        disjunction_certainty = disjunction.get("certainty", 1.0)
        negated_fact_certainty = negated_fact.get("certainty", 1.0)
        certainty = min(disjunction_certainty, negated_fact_certainty)
        
        # Create new disjunction concept
        new_disjunction = create_concept(
            identifier=new_disjunction_id,
            vector=new_disjunction_vector,
            metadata={
                "derived_from": [disjunction.get("identifier"), negated_fact.get("identifier")],
                "derivation_method": "disjunctive_syllogism",
                "disjunction_text": disjunction.get("metadata", {}).get("disjunction_text", ""),
                "negated_fact_text": negated_fact.get("metadata", {}).get("fact_text", ""),
                "disjuncts": remaining_disjuncts,
                "source": "derived",
                "derivation_pattern": "disjunctive_syllogism"
            },
            certainty=certainty
        )
        
        logger.info(f"Created new disjunction: {new_disjunction_id} with certainty {certainty}")
        return new_disjunction