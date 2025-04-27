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
    calculate_similarity, create_role_vectors, 
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
                     role_vectors: Dict[str, np.ndarray],
                     similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Apply modus ponens rule of inference: if P→Q and P, then Q.
    
    Args:
        rule (Dict[str, Any]): Rule representation (P→Q)
        fact (Dict[str, Any]): Fact representation (P)
        role_vectors (Dict[str, np.ndarray]): Role vectors for structured binding
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
    
    # Extract conditional parts using role vectors
    conditional_parts = extract_conditional_parts(rule_vector, role_vectors)
    antecedent_vector = conditional_parts["antecedent"]
    consequent_vector = conditional_parts["consequent"]
    
    # Check if fact matches antecedent using vector similarity
    similarity = calculate_similarity(fact_vector, antecedent_vector)
    
    if similarity < similarity_threshold:
        raise ValueError(f"Fact doesn't match rule antecedent (similarity: {similarity:.4f})")
    
    # Create a unique ID for the conclusion
    conclusion_id = str(uuid.uuid4())[:8]
    entity_id = fact["attributes"].get("entity_id", "")
    if entity_id:
        conclusion_id = f"{entity_id}_conclusion_{conclusion_id}"
    
    # Extract or create text representation for the conclusion
    conclusion_text = rule["attributes"].get("consequent", "")
    
    # Calculate certainty: min(certainty(rule), certainty(fact)) * similarity
    rule_certainty = rule["attributes"].get("certainty", 1.0)
    fact_certainty = fact["attributes"].get("certainty", 1.0)
    conclusion_certainty = min(rule_certainty, fact_certainty) * similarity
    
    # Create conclusion representation
    conclusion = {
        "identifier": conclusion_id,
        "type": "conclusion",
        "vector": consequent_vector,
        "attributes": {
            "text": conclusion_text,
            "certainty": conclusion_certainty,
            "derived_from": [rule["identifier"], fact["identifier"]],
            "derivation_pattern": "modus_ponens",
            "similarity": similarity,
            "entity_id": entity_id
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return conclusion

def apply_modus_tollens(rule: Dict[str, Any], negated_fact: Dict[str, Any], 
                       role_vectors: Dict[str, np.ndarray],
                       similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Apply modus tollens rule of inference: if P→Q and ¬Q, then ¬P.
    
    Args:
        rule (Dict[str, Any]): Rule representation (P→Q)
        negated_fact (Dict[str, Any]): Negated fact representation (¬Q)
        role_vectors (Dict[str, np.ndarray]): Role vectors for structured binding
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
    
    # Create a unique ID for the conclusion
    conclusion_id = str(uuid.uuid4())[:8]
    entity_id = negated_fact["attributes"].get("entity_id", "")
    if entity_id:
        conclusion_id = f"{entity_id}_conclusion_{conclusion_id}"
    
    # Calculate negation of antecedent vector
    # Simple approximation of negation
    negated_antecedent_vector = -antecedent_vector
    
    # Calculate certainty: min(certainty(rule), certainty(negated_fact)) * similarity
    rule_certainty = rule["attributes"].get("certainty", 1.0)
    fact_certainty = negated_fact["attributes"].get("certainty", 1.0)
    conclusion_certainty = min(rule_certainty, fact_certainty) * similarity
    
    # Create conclusion representation
    conclusion = {
        "identifier": conclusion_id,
        "type": "conclusion",
        "vector": negated_antecedent_vector,
        "attributes": {
            "text": f"Not {rule['attributes'].get('antecedent', '')}",
            "certainty": conclusion_certainty,
            "derived_from": [rule["identifier"], negated_fact["identifier"]],
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
    entity_id = fact_a["attributes"].get("entity_id") or fact_b["attributes"].get("entity_id", "")
    if entity_id:
        conjunction_id = f"{entity_id}_conjunction_{conjunction_id}"
    
    # Bundle the vectors to represent conjunction
    fact_a_vector = fact_a["vector"]
    fact_b_vector = fact_b["vector"]
    conjunction_vector = bundle_vectors([fact_a_vector, fact_b_vector])
    
    # Calculate certainty: min(certainty(P), certainty(Q))
    fact_a_certainty = fact_a["attributes"].get("certainty", 1.0)
    fact_b_certainty = fact_b["attributes"].get("certainty", 1.0)
    conjunction_certainty = min(fact_a_certainty, fact_b_certainty)
    
    # Extract text representations
    fact_a_text = fact_a["attributes"].get("text", "")
    fact_b_text = fact_b["attributes"].get("text", "")
    conjunction_text = f"{fact_a_text} and {fact_b_text}"
    
    # Create conjunction representation
    conjunction = {
        "identifier": conjunction_id,
        "type": "conjunction",
        "vector": conjunction_vector,
        "attributes": {
            "text": conjunction_text,
            "certainty": conjunction_certainty,
            "components": [fact_a["identifier"], fact_b["identifier"]],
            "derivation_pattern": "conjunction_introduction",
            "entity_id": entity_id
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return conjunction

def create_reasoning_chain(initial_premises: List[Dict[str, Any]], 
                         rules: List[Dict[str, Any]],
                         role_vectors: Dict[str, np.ndarray],
                         max_depth: int = 3,
                         similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Create a bounded reasoning chain by applying rules to premises.
    
    Args:
        initial_premises (List[Dict[str, Any]]): Initial facts/premises
        rules (List[Dict[str, Any]]): Rules to apply
        role_vectors (Dict[str, np.ndarray]): Role vectors for structural binding
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
                derivation_sig = f"{rule['identifier']}_{fact['identifier']}"
                if derivation_sig in used_derivations:
                    continue  # Skip if this derivation has been used before
                
                try:
                    # Try to apply modus ponens
                    conclusion = apply_modus_ponens(rule, fact, role_vectors, similarity_threshold)
                    
                    # Record the reasoning step
                    step = {
                        "step_id": len(reasoning_steps) + 1,
                        "pattern": "modus_ponens",
                        "premises": [rule["identifier"], fact["identifier"]],
                        "conclusion": conclusion["identifier"],
                        "certainty": conclusion["attributes"]["certainty"]
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
        if "derived_from" in fact["attributes"] and fact["identifier"] not in [
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
        conclusions, key=lambda c: c["attributes"].get("certainty", 0.0), reverse=True
    )
    
    # Introduce the main conclusion
    main_conclusion = sorted_conclusions[0]
    certainty = main_conclusion["attributes"].get("certainty", 0.0)
    certainty_percent = int(certainty * 100)
    
    explanation_lines.append(
        f"Based on the analysis, I've determined that {main_conclusion['attributes'].get('text', '')} "
        f"with {certainty_percent}% certainty."
    )
    
    # Add reasoning steps
    if reasoning_steps:
        explanation_lines.append("\nThis conclusion is based on the following reasoning:")
        
        for step in reasoning_steps:
            step_id = step["step_id"]
            pattern = step["pattern"]
            premises = step["premises"]
            conclusion_id = step["conclusion"]
            step_certainty = step.get("certainty", 0.0)
            step_certainty_percent = int(step_certainty * 100)
            
            # Get premise texts
            premise_texts = []
            for premise_id in premises:
                if premise_id in concepts:
                    premise_text = concepts[premise_id]["attributes"].get("text", premise_id)
                    premise_texts.append(premise_text)
                else:
                    premise_texts.append(premise_id)
            
            # Get conclusion text
            conclusion_text = conclusion_id
            if conclusion_id in concepts:
                conclusion_text = concepts[conclusion_id]["attributes"].get("text", conclusion_id)
            
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
                
            conclusion_text = conclusion["attributes"].get("text", "")
            conclusion_certainty = conclusion["attributes"].get("certainty", 0.0)
            conclusion_certainty_percent = int(conclusion_certainty * 100)
            
            explanation_lines.append(
                f"- {conclusion_text} ({conclusion_certainty_percent}% certainty)"
            )
    
    return "\n".join(explanation_lines)