"""
Ontology Mapper Module

This module handles mapping between natural language and domain-specific ontology terms
using pattern matching and LLM-based classification.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
import json
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

def create_ontology_mapper(domain_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new ontology mapper with the given domain configuration.
    
    Args:
        domain_config: Domain configuration containing the ontology
        
    Returns:
        Dict containing the initialized ontology mapper
    """
    domain = domain_config.get("domain", "general")
    ontology = domain_config.get("ontology", {})
        
    # Extract term mapping rules
    term_mapping_rules = domain_config.get("term_mapping_rules", [])
    compiled_rules = compile_regex_rules(term_mapping_rules)
    
    # Build reverse lookup for phrases to terms
    phrase_to_term_map = build_phrase_to_term_map(ontology)
    
    logger.info(f"Initialized ontology mapper for {domain} domain with {len(ontology)} categories")
    
    return {
        "domain": domain,
        "ontology": ontology,
        "term_mapping_rules": term_mapping_rules,
        "compiled_rules": compiled_rules,
        "phrase_to_term_map": phrase_to_term_map
    }

def compile_regex_rules(term_mapping_rules: List[Dict[str, str]]) -> List[Tuple[re.Pattern, str]]:
    """
    Compile regex patterns from mapping rules.
    
    Args:
        term_mapping_rules: List of rule dictionaries
        
    Returns:
        List of tuples containing compiled regex patterns and terms
    """
    compiled_rules = []
    
    for rule in term_mapping_rules:
        pattern = rule.get("regex", "")
        term = rule.get("ontology_term", "")
        
        if pattern and term:
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                compiled_rules.append((compiled_pattern, term))
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {str(e)}")
    
    return compiled_rules

def build_phrase_to_term_map(ontology: Dict[str, Dict[str, List[str]]]) -> Dict[str, str]:
    """
    Build a mapping from phrases to ontology terms.
    
    Args:
        ontology: The domain ontology structure
        
    Returns:
        Dictionary mapping phrases to ontology terms
    """
    phrase_map = {}
    
    for category, terms in ontology.items():
        for term, phrases in terms.items():
            for phrase in phrases:
                # Normalize the phrase (lowercase)
                norm_phrase = phrase.lower()
                phrase_map[norm_phrase] = term
    
    return phrase_map

@lru_cache(maxsize=1024)
def map_text_to_ontology(text: str, mapper: Dict[str, Any]) -> Tuple[str, float]:
    """
    Map natural language text to a standardized ontology term.
    
    Args:
        text: Natural language text to map
        mapper: The ontology mapper dictionary
        
    Returns:
        Tuple of (ontology_term, confidence)
    """
    # Try exact match in phrases
    text_lower = text.lower()
    phrase_to_term_map = mapper["phrase_to_term_map"]
    
    # Check if any phrases are contained in the text
    for phrase, term in phrase_to_term_map.items():
        if phrase in text_lower:
            confidence = 0.9  # High confidence for direct phrase match
            logger.debug(f"Phrase match: '{phrase}' → {term} (confidence: {confidence})")
            return term, confidence
    
    # Try regex pattern matching
    for pattern, term in mapper["compiled_rules"]:
        if pattern.search(text_lower):
            confidence = 0.85  # Slightly lower confidence for pattern match
            logger.debug(f"Pattern match: '{pattern.pattern}' → {term} (confidence: {confidence})")
            return term, confidence
    
    # No match found
    logger.debug(f"No ontology mapping found for: '{text}'")
    return "", 0.0

def map_text_to_ontology_with_llm(text: str, mapper: Dict[str, Any], 
                                  llm_interface: Any, llm_options: Dict[str, Any]) -> Tuple[str, float]:
    """
    Map text to ontology terms using LLM when direct mapping fails.
    
    Args:
        text: Text to map
        mapper: Ontology mapper
        llm_interface: LLM interface module
        llm_options: LLM options
        
    Returns:
        Tuple of (ontology_term, confidence)
    """
    # First try direct pattern matching
    term, confidence = map_text_to_ontology(text, mapper)
    
    # If direct matching found a term with reasonable confidence, use it
    if term and confidence >= 0.7:
        return term, confidence
    
    # Otherwise use LLM for mapping
    try:
        # Create a prompt for the LLM with the ontology structure
        ontology_json = json.dumps(mapper["ontology"], indent=2)
        
        prompt = f"""
        Map the following text to ONE of the standardized ontology terms in the {mapper["domain"]} domain:
        
        Text: "{text}"
        
        Ontology:
        {ontology_json}
        
        Return ONLY the most appropriate ontology term (e.g., PE_RATIO_LOW) that matches the text.
        If no term matches, return "NO_MATCH".
        """
        
        # Call the LLM
        response = llm_interface.call_openai_api(prompt, llm_options.get("model", "gpt-4"), llm_options)
        term = response.choices[0].message.content.strip()
        
        # Remove quotes if present
        term = term.strip('"\'')
        
        # Check if the term exists in the ontology
        term_exists = any(term in category_terms for category_terms in mapper["ontology"].values())
        
        if term == "NO_MATCH" or not term_exists:
            return "", 0.0
        
        # LLM classification has medium confidence
        return term, 0.75
        
    except Exception as e:
        logger.error(f"LLM mapping error: {str(e)}")
        return "", 0.0

def standardize_rule_text(rule_text: str, mapper: Dict[str, Any], 
                         llm_interface: Optional[Any] = None, 
                         llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert a natural language rule to a standardized form using ontology terms.
    
    Args:
        rule_text: Natural language rule text
        mapper: Ontology mapper
        llm_interface: Optional LLM interface for advanced mapping
        llm_options: Optional LLM options
        
    Returns:
        Dictionary with standardized rule components
    """
    # Extract conditional parts (if-then structure)
    if "if" in rule_text.lower() and "then" in rule_text.lower():
        parts = rule_text.lower().split("then")
        antecedent_text = parts[0].replace("if", "", 1).strip()
        consequent_text = parts[1].strip().rstrip(".")
        
        # Map antecedent and consequent to ontology terms
        if llm_interface and llm_options:
            antecedent_term, ant_confidence = map_text_to_ontology_with_llm(
                antecedent_text, mapper, llm_interface, llm_options)
            consequent_term, cons_confidence = map_text_to_ontology_with_llm(
                consequent_text, mapper, llm_interface, llm_options)
        else:
            antecedent_term, ant_confidence = map_text_to_ontology(antecedent_text, mapper)
            consequent_term, cons_confidence = map_text_to_ontology(consequent_text, mapper)
        
        # Create standardized rule representation
        standardized_rule = {
            "original_text": rule_text,
            "is_conditional": True,
            "antecedent": {
                "text": antecedent_text,
                "term": antecedent_term if antecedent_term else "UNKNOWN",
                "confidence": ant_confidence
            },
            "consequent": {
                "text": consequent_text,
                "term": consequent_term if consequent_term else "UNKNOWN",
                "confidence": cons_confidence
            }
        }
    else:
        # Non-conditional statement
        if llm_interface and llm_options:
            term, confidence = map_text_to_ontology_with_llm(rule_text, mapper, llm_interface, llm_options)
        else:
            term, confidence = map_text_to_ontology(rule_text, mapper)
            
        standardized_rule = {
            "original_text": rule_text,
            "is_conditional": False,
            "term": term if term else "UNKNOWN",
            "confidence": confidence
        }
    
    return standardized_rule

def standardize_fact_text(fact_text: str, mapper: Dict[str, Any],
                         llm_interface: Optional[Any] = None,
                         llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert a natural language fact to a standardized form using ontology terms.
    
    Args:
        fact_text: Natural language fact text
        mapper: Ontology mapper
        llm_interface: Optional LLM interface for advanced mapping
        llm_options: Optional LLM options
        
    Returns:
        Dictionary with standardized fact components
    """
    # Map fact to ontology term
    if llm_interface and llm_options:
        term, confidence = map_text_to_ontology_with_llm(fact_text, mapper, llm_interface, llm_options)
    else:
        term, confidence = map_text_to_ontology(fact_text, mapper)
    
    # Create standardized fact representation
    standardized_fact = {
        "original_text": fact_text,
        "term": term if term else "UNKNOWN",
        "confidence": confidence
    }
    
    return standardized_fact

def extract_ontology_terms(text: str, mapper: Dict[str, Any]) -> List[str]:
    """
    Extract all matching ontology terms from a text.
    
    Args:
        text: Text to analyze
        mapper: Ontology mapper
        
    Returns:
        List of ontology terms found in the text
    """
    found_terms = []
    text_lower = text.lower()
    
    # Check for phrase matches
    for phrase, term in mapper["phrase_to_term_map"].items():
        if phrase in text_lower and term not in found_terms:
            found_terms.append(term)
    
    # Check for regex pattern matches
    for pattern, term in mapper["compiled_rules"]:
        if pattern.search(text_lower) and term not in found_terms:
            found_terms.append(term)
    
    return found_terms