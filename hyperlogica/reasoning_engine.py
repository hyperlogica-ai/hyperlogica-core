"""
Reasoning Engine Module

Pure functional implementation of the hyperdimensional reasoning engine for Hyperlogica.
Implements bounded syllogistic reasoning with controlled uncertainty propagation.
"""

import logging
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from datetime import datetime

# Import vector operations
from .vector_operations import (
    normalize_vector, bind_vectors, unbind_vectors, bundle_vectors, 
    calculate_similarity, create_role_vectors, generate_vector,
    create_conditional_representation, extract_conditional_parts
)

# Configure logging
logger = logging.getLogger(__name__)

def create_concept(identifier: str, vector: np.ndarray, 
                  attributes: Dict[str, Any], 
                  concept_type: str = "concept") -> Dict[str, Any]:
    """
    Create a concept representation with vector and attributes.
    
    Args:
        identifier (str): Unique identifier for the concept
        vector (np.ndarray): Vector representation for the concept
        attributes (Dict[str, Any]): Attributes including certainty, text, etc.
        concept_type (str): Type of the concept ("concept", "rule", "fact", etc.)
        
    Returns:
        Dict[str, Any]: The concept representation
    """
    # Ensure the concept has a certainty attribute
    if "certainty" not in attributes:
        attributes["certainty"] = 1.0  # Default high certainty
    
    # Create the concept structure
    concept = {
        "identifier": identifier,
        "type": concept_type,
        "vector": vector,
        "attributes": attributes,
        "timestamp": datetime.now().isoformat()
    }
    
    return concept

def parse_rule(rule_text: str) -> Tuple[str, str]:
    """
    Parse a rule text into antecedent and consequent components.
    
    Args:
        rule_text (str): Natural language rule text
        
    Returns:
        Tuple[str, str]: (antecedent, consequent) parts of the rule
        
    Raises:
        ValueError: If the rule cannot be parsed
    """
    # Try to split on "then"
    if "then" in rule_text.lower():
        parts = rule_text.lower().split("then")
        antecedent = parts[0].strip()
        
        # Remove "if" from the beginning if present
        if antecedent.startswith("if "):
            antecedent = antecedent[3:].strip()
            
        consequent = parts[1].strip().rstrip(".")
        return antecedent, consequent
    
    # Try to split on "implies", "results in", etc.
    implies_terms = ["implies", "leads to", "results in", "causes"]
    for term in implies_terms:
        if term in rule_text.lower():
            parts = rule_text.lower().split(term)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    
    # Not a recognizable rule format
    raise ValueError(f"Could not parse rule: {rule_text}")

def is_conditional(concept: Dict[str, Any]) -> bool:
    """
    Check if a concept is a conditional rule.
    
    Args:
        concept (Dict[str, Any]): Concept to check
        
    Returns:
        bool: True if the concept is a conditional rule
    """
    attributes = concept.get("attributes", {})
    
    # Check for explicit marking
    if attributes.get("conditional", False):
        return True
    
    # Check for antecedent/consequent attributes
    if "antecedent" in attributes and "consequent" in attributes:
        return True
    
    # Check type
    if concept.get("type") == "rule":
        return True
    
    # Check identifier pattern
    if "_if_" in concept.get("identifier", ""):
        return True
    
    # Not a conditional
    return False

def extract_antecedent(rule: Dict[str, Any]) -> str:
    """
    Extract the antecedent (if-part) from a rule.
    
    Args:
        rule (Dict[str, Any]): Rule representation
        
    Returns:
        str: The antecedent text
        
    Raises:
        ValueError: If rule is not conditional or antecedent cannot be extracted
    """
    if not is_conditional(rule):
        raise ValueError("Cannot extract antecedent from non-conditional rule")
    
    # Check attributes for explicit antecedent
    attributes = rule.get("attributes", {})
    if "antecedent" in attributes:
        return attributes["antecedent"]
    
    # Check if rule ID has _if_ format
    identifier = rule.get("identifier", "")
    if "_if_" in identifier:
        parts = identifier.split("_if_")
        if len(parts) == 2:
            return parts[1]
    
    # Check for rule_text and try to parse it
    if "rule_text" in attributes:
        rule_text = attributes["rule_text"]
        try:
            antecedent, _ = parse_rule(rule_text)
            return antecedent
        except ValueError:
            pass
    
    raise ValueError(f"Could not extract antecedent from rule: {rule.get('identifier', 'unknown')}")

def extract_consequent(rule: Dict[str, Any]) -> str:
    """
    Extract the consequent (then-part) from a rule.
    
    Args:
        rule (Dict[str, Any]): Rule representation
        
    Returns:
        str: The consequent text
        
    Raises:
        ValueError: If rule is not conditional or consequent cannot be extracted
    """
    if not is_conditional(rule):
        raise ValueError("Cannot extract consequent from non-conditional rule")
    
    # Check attributes for explicit consequent
    attributes = rule.get("attributes", {})
    if "consequent" in attributes:
        return attributes["consequent"]
    
    # Check if rule ID has _if_ format
    identifier = rule.get("identifier", "")
    if "_if_" in identifier:
        parts = identifier.split("_if_")
        if len(parts) == 2:
            return parts[0]
    
    # Check for rule_text and try to parse it
    if "rule_text" in attributes:
        rule_text = attributes["rule_text"]
        try:
            _, consequent = parse_rule(rule_text)
            return consequent
        except ValueError:
            pass
    
    raise ValueError(f"Could not extract consequent from rule: {rule.get('identifier', 'unknown')}")

def cleanse_vector(vector: np.ndarray) -> np.ndarray:
    """
    Clean up a vector by removing NaN/Inf values and normalizing.
    
    Args:
        vector (np.ndarray): Vector to cleanse
        
    Returns:
        np.ndarray: Cleansed vector
    """
    # Replace any NaN/Inf values with zeros
    vector = np.nan_to_num(vector)
    
    # Normalize the vector
    return normalize_vector(vector)

def matches(fact: Dict[str, Any], pattern: str, store: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Check if a fact matches a pattern based on text similarity or vector similarity.
    
    Args:
        fact (Dict[str, Any]): Fact representation to check
        pattern (str): Pattern to match against
        store (Dict[str, Any]): Vector store for similarity search
        
    Returns:
        Tuple[bool, float]: (match_result, similarity_score)
    """
    # Default similarity threshold
    threshold = 0.65
    
    # Get fact content for comparison
    fact_text = fact.get("attributes", {}).get("text", "")
    if not fact_text:
        fact_text = fact.get("identifier", "")
    
    # Simple text-based matching
    fact_text_lower = fact_text.lower()
    pattern_lower = pattern.lower()
    
    # Direct substring match
    if pattern_lower in fact_text_lower:
        # Calculate a similarity score based on relative length of match
        similarity = len(pattern_lower) / max(len(fact_text_lower), 1)
        # Ensure similarity is at least threshold
        similarity = max(threshold + 0.1, similarity)
        return True, similarity
    
    # Check if vectors are available for similarity calculation
    if "vector" in fact and store and "dimension" in store:
        # Generate pattern vector using the same dimension as the store
        try:
            dimension = store.get("dimension", 1000)
            pattern_vector = generate_vector(pattern, dimension)
            fact_vector = fact["vector"]
            
            # Calculate similarity
            similarity = calculate_similarity(pattern_vector, fact_vector)
            
            # Check against threshold
            if similarity >= threshold:
                return True, similarity
        except Exception as e:
            logger.warning(f"Error in vector similarity calculation: {str(e)}")
    
    # No match found
    return False, 0.0

def calculate_certainty(evidence_certainties: List[float], method: str = "min") -> float:
    """
    Calculate overall certainty from multiple pieces of evidence.
    
    Args:
        evidence_certainties (List[float]): List of certainty values for different evidence
        method (str): Method for combining certainties ("min", "product", "noisy_or", "weighted")
        
    Returns:
        float: Combined certainty value between 0 and 1
        
    Raises:
        ValueError: If evidence_certainties is empty or method is invalid
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
        context (Dict[str, Any]): Contextual information for calibration
        method (str): Recalibration method ("linear", "sigmoid", "expert", "historical")
        
    Returns:
        float: Recalibrated certainty value between 0 and 1
        
    Raises:
        ValueError: If raw_certainty is outside [0,1] or method is invalid
    """
    if not (0 <= raw_certainty <= 1):
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
        import math
        steepness = context.get("steepness", 10.0)  # Controls how sharp the sigmoid is
        midpoint = context.get("midpoint", 0.5)  # Value where sigmoid equals 0.5
        
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

def apply_modus_ponens(rule: Dict[str, Any], fact: Dict[str, Any], 
                     role_vectors: Optional[Dict[str, np.ndarray]] = None,
                     similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Apply modus ponens rule of inference: if P→Q and P, then Q.
    
    Args:
        rule (Dict[str, Any]): Rule representation (P→Q)
        fact (Dict[str, Any]): Fact representation (P)
        role_vectors (Dict[str, np.ndarray], optional): Role vectors for structured binding
        similarity_threshold (float): Threshold for considering a match
        
    Returns:
        Dict[str, Any]: Derived conclusion (Q) with certainty
        
    Raises:
        ValueError: If rule is not conditional or fact doesn't match antecedent
    """
    # Check if rule is conditional
    if not is_conditional(rule):
        raise ValueError("Rule must be conditional for modus ponens")
    
    # Extract components
    rule_vector = rule["vector"]
    fact_vector = fact["vector"]
    
    # Check if we have role vectors for structured extraction
    if role_vectors is not None and "antecedent" in role_vectors and "consequent" in role_vectors:
        # Extract conditional parts using role vectors
        conditional_parts = extract_conditional_parts(rule_vector, role_vectors)
        antecedent_vector = conditional_parts["antecedent"]
        consequent_vector = conditional_parts["consequent"]
        
        # Check if fact matches antecedent using vector similarity
        similarity = calculate_similarity(fact_vector, antecedent_vector)
        
        if similarity < similarity_threshold:
            raise ValueError(f"Fact doesn't match rule antecedent (similarity: {similarity:.4f})")
    else:
        # Extract antecedent text for simpler matching
        try:
            antecedent_text = extract_antecedent(rule)
            match_result, similarity = matches(fact, antecedent_text, None)
            
            if not match_result:
                raise ValueError(f"Fact doesn't match rule antecedent (similarity: {similarity:.4f})")
                
            # Use simple unbinding to estimate consequent vector
            consequent_vector = unbind_vectors(rule_vector, fact_vector)
            consequent_vector = cleanse_vector(consequent_vector)
        except ValueError as e:
            raise ValueError(f"Error applying modus ponens: {str(e)}")
    
    # Create a unique ID for the conclusion
    conclusion_id = str(uuid.uuid4())[:8]
    entity_id = fact.get("attributes", {}).get("entity_id", "")
    if entity_id:
        conclusion_id = f"{entity_id}_conclusion_{conclusion_id}"
    
    # Extract or create text representation for the conclusion
    try:
        conclusion_text = extract_consequent(rule)
    except ValueError:
        conclusion_text = f"Conclusion from {rule.get('identifier', 'unknown_rule')}"
    
    # Calculate certainty: min(certainty(rule), certainty(fact)) * similarity
    rule_certainty = rule.get("attributes", {}).get("certainty", 1.0)
    fact_certainty = fact.get("attributes", {}).get("certainty", 1.0)
    conclusion_certainty = min(rule_certainty, fact_certainty) * similarity
    
    # Create conclusion representation
    conclusion = {
        "identifier": conclusion_id,
        "type": "conclusion",
        "vector": consequent_vector,
        "attributes": {
            "text": conclusion_text,
            "certainty": conclusion_certainty,
            "derived_from": [rule.get("identifier", "unknown"), fact.get("identifier", "unknown")],
            "derivation_pattern": "modus_ponens",
            "similarity": similarity,
            "entity_id": entity_id
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return conclusion

def apply_modus_tollens(rule: Dict[str, Any], negated_fact: Dict[str, Any], 
                       role_vectors: Optional[Dict[str, np.ndarray]] = None,
                       similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Apply modus tollens rule of inference: if P→Q and ¬Q, then ¬P.
    
    Args:
        rule (Dict[str, Any]): Rule representation (P→Q)
        negated_fact (Dict[str, Any]): Negated fact representation (¬Q)
        role_vectors (Dict[str, np.ndarray], optional): Role vectors for structured binding
        similarity_threshold (float): Threshold for considering a match
        
    Returns:
        Dict[str, Any]: Derived conclusion (¬P) with certainty
        
    Raises:
        ValueError: If rule is not conditional or negated fact doesn't match consequent
    """
    # Check if rule is conditional
    if not is_conditional(rule):
        raise ValueError("Rule must be conditional for modus tollens")
    
    # Extract components
    rule_vector = rule["vector"]
    negated_fact_vector = negated_fact["vector"]
    
    # Check if we have role vectors for structured extraction
    if role_vectors is not None and "antecedent" in role_vectors and "consequent" in role_vectors:
        # Extract conditional parts using role vectors
        conditional_parts = extract_conditional_parts(rule_vector, role_vectors)
        antecedent_vector = conditional_parts["antecedent"]
        consequent_vector = conditional_parts["consequent"]
        
        # Check if negated fact matches negation of consequent
        # For simple vector representations, we can approximate negation by calculating
        # similarity to the complement of the consequent vector
        complementary_consequent = -consequent_vector  # Simple approximation of negation
        similarity = calculate_similarity(negated_fact_vector, complementary_consequent)
        
        if similarity < similarity_threshold:
            raise ValueError(f"Negated fact doesn't match negation of consequent (similarity: {similarity:.4f})")
            
        # Calculate negation of antecedent vector
        # Simple approximation of negation
        negated_antecedent_vector = -antecedent_vector
    else:
        # Extract consequent text for simpler matching
        try:
            consequent_text = extract_consequent(rule)
            
            # Check if negated fact explicitly states negation
            fact_text = negated_fact.get("attributes", {}).get("text", "").lower()
            
            # Simple heuristic for detecting negation in text
            negation_terms = ["not", "isn't", "doesn't", "don't", "no ", "never"]
            is_negated = any(term in fact_text for term in negation_terms)
            
            if not is_negated:
                raise ValueError("Fact does not appear to be a negation")
            
            # Check that the negated fact relates to the consequent
            match_result, similarity = matches(negated_fact, consequent_text, None)
            
            if not match_result:
                raise ValueError(f"Negated fact doesn't match rule consequent (similarity: {similarity:.4f})")
                
            # Use vector operations to generate the negated antecedent
            antecedent_text = extract_antecedent(rule)
            dimension = len(rule_vector)
            antecedent_vector = generate_vector(antecedent_text, dimension)
            negated_antecedent_vector = -antecedent_vector  # Simple negation
        except ValueError as e:
            raise ValueError(f"Error applying modus tollens: {str(e)}")
    
    # Create a unique ID for the conclusion
    conclusion_id = str(uuid.uuid4())[:8]
    entity_id = negated_fact.get("attributes", {}).get("entity_id", "")
    if entity_id:
        conclusion_id = f"{entity_id}_conclusion_{conclusion_id}"
    
    # Calculate certainty: min(certainty(rule), certainty(negated_fact)) * similarity
    rule_certainty = rule.get("attributes", {}).get("certainty", 1.0)
    fact_certainty = negated_fact.get("attributes", {}).get("certainty", 1.0)
    conclusion_certainty = min(rule_certainty, fact_certainty) * similarity
    
    # Generate text representation for the conclusion
    try:
        antecedent_text = extract_antecedent(rule)
        conclusion_text = f"Not {antecedent_text}"
    except ValueError:
        conclusion_text = f"Negated conclusion from {rule.get('identifier', 'unknown_rule')}"
    
    # Create conclusion representation
    conclusion = {
        "identifier": conclusion_id,
        "type": "conclusion",
        "vector": negated_antecedent_vector,
        "attributes": {
            "text": conclusion_text,
            "certainty": conclusion_certainty,
            "derived_from": [rule.get("identifier", "unknown"), negated_fact.get("identifier", "unknown")],
            "derivation_pattern": "modus_tollens",
            "similarity": similarity,
            "entity_id": entity_id
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return conclusion

def apply_conjunction_introduction(fact_a: Dict[str, Any], fact_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply conjunction introduction: P, Q, therefore P∧Q.
    
    Args:
        fact_a (Dict[str, Any]): First fact representation (P)
        fact_b (Dict[str, Any]): Second fact representation (Q)
        
    Returns:
        Dict[str, Any]: Conjunction representation (P∧Q) with certainty
    """
    # Create a unique ID for the conjunction
    conjunction_id = str(uuid.uuid4())[:8]
    entity_id = fact_a.get("attributes", {}).get("entity_id", "") or fact_b.get("attributes", {}).get("entity_id", "")
    if entity_id:
        conjunction_id = f"{entity_id}_conjunction_{conjunction_id}"
    
    # Bundle the vectors to represent conjunction
    fact_a_vector = fact_a["vector"]
    fact_b_vector = fact_b["vector"]
    conjunction_vector = bundle_vectors([fact_a_vector, fact_b_vector])
    
    # Calculate certainty: min(certainty(P), certainty(Q))
    fact_a_certainty = fact_a.get("attributes", {}).get("certainty", 1.0)
    fact_b_certainty = fact_b.get("attributes", {}).get("certainty", 1.0)
    conjunction_certainty = min(fact_a_certainty, fact_b_certainty)
    
    # Extract text representations
    fact_a_text = fact_a.get("attributes", {}).get("text", fact_a.get("identifier", ""))
    fact_b_text = fact_b.get("attributes", {}).get("text", fact_b.get("identifier", ""))
    conjunction_text = f"{fact_a_text} and {fact_b_text}"
    
    # Create conjunction representation
    conjunction = {
        "identifier": conjunction_id,
        "type": "conjunction",
        "vector": conjunction_vector,
        "attributes": {
            "text": conjunction_text,
            "certainty": conjunction_certainty,
            "components": [fact_a.get("identifier", "unknown"), fact_b.get("identifier", "unknown")],
            "derivation_pattern": "conjunction_introduction",
            "entity_id": entity_id
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return conjunction

def create_reasoning_chain(initial_premises: List[Dict[str, Any]], 
                         rules: List[Dict[str, Any]],
                         role_vectors: Optional[Dict[str, np.ndarray]] = None,
                         max_depth: int = 3,
                         similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Create a bounded reasoning chain by applying rules to premises.
    
    Args:
        initial_premises (List[Dict[str, Any]]): Initial facts/premises
        rules (List[Dict[str, Any]]): Rules to apply
        role_vectors (Dict[str, np.ndarray], optional): Role vectors for structural binding
        max_depth (int): Maximum reasoning depth to prevent infinite chains
        similarity_threshold (float): Threshold for considering a match
        
    Returns:
        Dict[str, Any]: Reasoning result with conclusions and trace
    """
    # Initialize reasoning state
    depth = 0
    derived_facts = initial_premises.copy()  # Start with initial premises
    reasoning_steps = []  # Track reasoning steps
    used_derivations = set()  # Track which derivations have been used (to prevent cycles)
    
    # Continue deriving until we reach max depth or no new facts can be derived
    while depth < max_depth:
        new_facts = []  # New facts derived in this iteration
        
        # Try to apply each rule to each fact
        for rule in rules:
            if not is_conditional(rule):
                continue  # Skip non-conditional rules
                
            for fact in derived_facts:
                # Create a derivation signature to prevent cycles
                derivation_sig = f"{rule.get('identifier', '')}_{fact.get('identifier', '')}"
                if derivation_sig in used_derivations:
                    continue  # Skip if this derivation has been used before
                
                try:
                    # Try to apply modus ponens
                    conclusion = apply_modus_ponens(rule, fact, role_vectors, similarity_threshold)
                    
                    # Record the reasoning step
                    step = {
                        "step_id": len(reasoning_steps) + 1,
                        "pattern": "modus_ponens",
                        "premises": [rule.get("identifier", "unknown"), fact.get("identifier", "unknown")],
                        "conclusion": conclusion["identifier"],
                        "certainty": conclusion.get("attributes", {}).get("certainty", 0.0)
                    }
                    reasoning_steps.append(step)
                    
                    # Add to new facts and mark derivation as used
                    new_facts.append(conclusion)
                    used_derivations.add(derivation_sig)
                except ValueError:
                    # Rule doesn't apply, continue to next fact
                    pass
        
        # If no new facts were derived, we've reached a fixed point
        if not new_facts:
            break
            
        # Add new facts to the set of derived facts for next iteration
        derived_facts.extend(new_facts)
        depth += 1
    
    # Prepare the result
    final_conclusions = []
    
    # Extract final conclusions (those that weren't used as premises for further derivation)
    for fact in derived_facts:
        # A fact is a final conclusion if it was derived (not an initial premise)
        # and it wasn't used as a premise for further derivation
        if "attributes" in fact and "derived_from" in fact["attributes"] and fact.get("identifier", "") not in [
            premise for step in reasoning_steps for premise in step["premises"]
        ]:
            final_conclusions.append(fact)
    
    # Create reasoning result
    result = {
        "depth": depth,
        "steps": reasoning_steps,
        "conclusions": final_conclusions,
        "all_derived_facts": derived_facts,
        "termination_reason": "fixed_point" if depth < max_depth else "max_depth_reached"
    }
    
    return result

def classify_conclusion(conclusion: Dict[str, Any], signal_vectors: Dict[str, np.ndarray]) -> str:
    """
    Classify a conclusion as positive, negative, or neutral based on vector similarity.
    
    Args:
        conclusion (Dict[str, Any]): Conclusion to classify
        signal_vectors (Dict[str, np.ndarray]): Reference vectors for signal types
        
    Returns:
        str: Signal type ("positive", "negative", or "neutral")
    """
    if "vector" not in conclusion:
        return "neutral"  # No vector to classify
        
    # Calculate similarity to each signal vector
    conclusion_vector = conclusion["vector"]
    similarities = {}
    
    for signal_type, signal_vector in signal_vectors.items():
        similarity = calculate_similarity(conclusion_vector, signal_vector)
        similarities[signal_type] = similarity
    
    # Find the most similar signal type
    max_signal = max(similarities.items(), key=lambda x: x[1])
    
    # Map signal vector names to standard output types
    signal_map = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "vector_positive": "positive",
        "vector_negative": "negative",
        "vector_neutral": "neutral"
    }
    
    return signal_map.get(max_signal[0], "neutral")

def generate_explanation(conclusions: List[Dict[str, Any]], 
                       reasoning_steps: List[Dict[str, Any]],
                       concepts: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a natural language explanation of the reasoning process.
    
    Args:
        conclusions (List[Dict[str, Any]]): Final conclusions
        reasoning_steps (List[Dict[str, Any]]): Steps in the reasoning process
        concepts (Dict[str, Dict[str, Any]]): Dictionary of all concepts by identifier
        
    Returns:
        str: Natural language explanation
    """
    # Start with an introduction based on the conclusions
    explanation_lines = []
    
    if not conclusions:
        return "No conclusions were derived from the available information."
    
    # Sort conclusions by certainty (highest first)
    sorted_conclusions = sorted(
        conclusions, key=lambda c: c.get("attributes", {}).get("certainty", 0.0), reverse=True
    )
    
    # Introduce the main conclusion
    main_conclusion = sorted_conclusions[0]
    certainty = main_conclusion.get("attributes", {}).get("certainty", 0.0)
    certainty_percent = int(certainty * 100)
    
    explanation_lines.append(
        f"Based on the analysis, I've determined that {main_conclusion.get('attributes', {}).get('text', '')} "
        f"with {certainty_percent}% certainty."
    )
    
    # Add reasoning steps
    if reasoning_steps:
        explanation_lines.append("\nThis conclusion is based on the following reasoning:")
        
        for step in reasoning_steps:
            step_id = step.get("step_id", 0)
            pattern = step.get("pattern", "")
            premises = step.get("premises", [])
            conclusion_id = step.get("conclusion", "")
            step_certainty = step.get("certainty", 0.0)
            step_certainty_percent = int(step_certainty * 100)
            
            # Get premise texts
            premise_texts = []
            for premise_id in premises:
                if premise_id in concepts:
                    premise_text = concepts[premise_id].get("attributes", {}).get("text", premise_id)
                    premise_texts.append(premise_text)
                else:
                    premise_texts.append(premise_id)
            
            # Get conclusion text
            conclusion_text = conclusion_id
            if conclusion_id in concepts:
                conclusion_text = concepts[conclusion_id].get("attributes", {}).get("text", conclusion_id)
            
            # Format based on pattern
            if pattern == "modus_ponens":
                explanation_lines.append(
                    f"Step {step_id}: Since \"{premise_texts[0]}\" and \"{premise_texts[1]}\", "
                    f"I can conclude that \"{conclusion_text}\" ({step_certainty_percent}% certainty)."
                )
            elif pattern == "conjunction_introduction":
                explanation_lines.append(
                    f"Step {step_id}: Combining \"{premise_texts[0]}\" and \"{premise_texts[1]}\", "
                    f"I can state that \"{conclusion_text}\" ({step_certainty_percent}% certainty)."
                )
            elif pattern == "modus_tollens":
                explanation_lines.append(
                    f"Step {step_id}: Since \"{premise_texts[0]}\" and not \"{premise_texts[1]}\", "
                    f"I can conclude that not \"{conclusion_text}\" ({step_certainty_percent}% certainty)."
                )
            else:
                explanation_lines.append(
                    f"Step {step_id}: From {', '.join(premise_texts)}, "
                    f"I derived \"{conclusion_text}\" ({step_certainty_percent}% certainty)."
                )
    
    # Add additional conclusions if there are multiple
    if len(sorted_conclusions) > 1:
        explanation_lines.append("\nAdditional conclusions:")
        
        for i, conclusion in enumerate(sorted_conclusions[1:], 1):
            if i >= 3:  # Limit to top 3 additional conclusions
                break
                
            conclusion_text = conclusion.get("attributes", {}).get("text", "")
            conclusion_certainty = conclusion.get("attributes", {}).get("certainty", 0.0)
            conclusion_certainty_percent = int(conclusion_certainty * 100)
            
            explanation_lines.append(
                f"- {conclusion_text} ({conclusion_certainty_percent}% certainty)"
            )
    
    return "\n".join(explanation_lines)

def record_reasoning_step(pattern: str, premises: List[str], conclusion: str, 
                        certainty: float, step_id: int) -> Dict[str, Any]:
    """
    Record a reasoning step for tracing.
    
    Args:
        pattern (str): Reasoning pattern applied (e.g., "modus_ponens")
        premises (List[str]): Identifiers of premise concepts
        conclusion (str): Identifier of conclusion concept
        certainty (float): Certainty of the conclusion
        step_id (int): Sequential step ID
        
    Returns:
        Dict[str, Any]: Dictionary recording the reasoning step
    """
    return {
        "step_id": step_id,
        "pattern": pattern,
        "premises": premises,
        "conclusion": conclusion,
        "certainty": certainty,
        "timestamp": datetime.now().isoformat()
    }

def explain_reasoning(concept: Dict[str, Any], reasoning_steps: List[Dict[str, Any]], 
                    store: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an explanation of how a concept was derived.
    
    Args:
        concept (Dict[str, Any]): Concept to explain
        reasoning_steps (List[Dict[str, Any]]): Available reasoning steps
        store (Dict[str, Any]): Vector store containing relevant concepts
        
    Returns:
        Dict[str, Any]: Explanation of the reasoning process
    """
    concept_id = concept.get("identifier", "")
    
    # Check if this is a base concept (not derived)
    if "attributes" not in concept or "derived_from" not in concept["attributes"]:
        return {
            "type": "base_concept",
            "identifier": concept_id,
            "certainty": concept.get("attributes", {}).get("certainty", 1.0),
            "explanation": f"This is a base concept: {concept.get('attributes', {}).get('text', concept_id)}"
        }
    
    # Find all steps that led to this conclusion
    concept_steps = []
    for step in reasoning_steps:
        if step.get("conclusion") == concept_id:
            concept_steps.append(step)
    
    if not concept_steps:
        return {
            "type": "unknown_derivation",
            "identifier": concept_id,
            "certainty": concept.get("attributes", {}).get("certainty", 0.0),
            "explanation": "Derivation steps not found for this concept"
        }
    
    # Get the most direct step for explaining
    step = concept_steps[0]  # Default to first step
    pattern = step.get("pattern", "unknown")
    premises = step.get("premises", [])
    certainty = step.get("certainty", 0.0)
    
    # Get premise concepts
    premise_concepts = []
    for premise_id in premises:
        if premise_id in store.get("concepts", {}):
            premise_concepts.append(store["concepts"][premise_id])
        else:
            # Create a placeholder for missing concepts
            premise_concepts.append({
                "identifier": premise_id,
                "attributes": {"text": premise_id}
            })
    
    # Structure the explanation based on pattern
    if pattern == "modus_ponens":
        rule_concept = premise_concepts[0] if premises else None
        fact_concept = premise_concepts[1] if len(premises) > 1 else None
        
        rule_text = rule_concept.get("attributes", {}).get("text", rule_concept.get("identifier", "")) if rule_concept else "unknown rule"
        fact_text = fact_concept.get("attributes", {}).get("text", fact_concept.get("identifier", "")) if fact_concept else "unknown fact"
        conclusion_text = concept.get("attributes", {}).get("text", concept_id)
        
        explanation = f"Applied modus ponens using the rule '{rule_text}' and the fact '{fact_text}' to derive '{conclusion_text}' with {certainty:.2f} certainty."
    
    elif pattern == "conjunction_introduction":
        fact_a = premise_concepts[0] if premises else None
        fact_b = premise_concepts[1] if len(premises) > 1 else None
        
        fact_a_text = fact_a.get("attributes", {}).get("text", fact_a.get("identifier", "")) if fact_a else "unknown fact"
        fact_b_text = fact_b.get("attributes", {}).get("text", fact_b.get("identifier", "")) if fact_b else "unknown fact"
        conclusion_text = concept.get("attributes", {}).get("text", concept_id)
        
        explanation = f"Combined '{fact_a_text}' and '{fact_b_text}' to form the conjunction '{conclusion_text}' with {certainty:.2f} certainty."
    
    elif pattern == "modus_tollens":
        rule_concept = premise_concepts[0] if premises else None
        negated_fact = premise_concepts[1] if len(premises) > 1 else None
        
        rule_text = rule_concept.get("attributes", {}).get("text", rule_concept.get("identifier", "")) if rule_concept else "unknown rule"
        negated_text = negated_fact.get("attributes", {}).get("text", negated_fact.get("identifier", "")) if negated_fact else "unknown negation"
        conclusion_text = concept.get("attributes", {}).get("text", concept_id)
        
        explanation = f"Applied modus tollens using the rule '{rule_text}' and the negated fact '{negated_text}' to derive '{conclusion_text}' with {certainty:.2f} certainty."
    
    else:
        premise_texts = [p.get("attributes", {}).get("text", p.get("identifier", "unknown")) for p in premise_concepts]
        conclusion_text = concept.get("attributes", {}).get("text", concept_id)
        
        explanation = f"Used {pattern} reasoning with premises: {', '.join(premise_texts)} to derive '{conclusion_text}' with {certainty:.2f} certainty."
    
    # Create the detailed explanation
    return {
        "type": "reasoning_chain",
        "identifier": concept_id,
        "pattern": pattern,
        "premises": premises,
        "final_certainty": certainty,
        "steps": concept_steps,
        "explanation": explanation
    }

def format_explanation(explanation: Dict[str, Any]) -> str:
    """
    Format an explanation as user-friendly text.
    
    Args:
        explanation (Dict[str, Any]): Explanation dictionary
        
    Returns:
        str: Formatted explanation text
    """
    if explanation["type"] == "base_concept":
        return f"This is a base concept or fact with {explanation.get('certainty', 1.0):.2f} certainty."
    
    elif explanation["type"] == "unknown_derivation":
        return "The derivation of this concept is not recorded in the reasoning trace."
    
    elif explanation["type"] == "reasoning_chain":
        explanation_text = explanation.get("explanation", "")
        
        # Add any extra steps
        if len(explanation.get("steps", [])) > 1:
            step_count = len(explanation["steps"])
            explanation_text += f"\n\nThis conclusion involved {step_count} reasoning steps in total."
        
        return explanation_text
    
    else:
        return "No explanation available for this concept."
    