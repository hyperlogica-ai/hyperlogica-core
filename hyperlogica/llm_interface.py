"""
Enhanced LLM Interface Module for Hyperlogica with Ontology Support

Adds ontology-aware text processing to the existing LLM interface module.
"""

import os
import logging
import json
import backoff
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI

# Import the ontology mapper
from .ontology_mapper import (
    create_ontology_mapper, 
    map_text_to_ontology,
    map_text_to_ontology_with_llm,
    standardize_rule_text,
    standardize_fact_text
)

# Configure logging
logger = logging.getLogger(__name__)

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def call_openai_api(prompt: str, model: str, options: Dict[str, Any]) -> Any:
    """
    Call the OpenAI API with backoff for rate limiting and error handling.
    
    Args:
        prompt: The text prompt to send
        model: The model to use (e.g., "gpt-4")
        options: Additional options for the API call
        
    Returns:
        The API response
    """
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Configure API call
    temperature = options.get("temperature", 0.0)
    max_tokens = options.get("max_tokens", 2000)
    
    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in AI knowledge representation and vector symbolic architectures."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response

def call_openai_api_cached(prompt: str, model: str, cache_file: str = None, options: Dict[str, Any] = None) -> Any:
    """
    Call the OpenAI API with caching to avoid redundant calls.
    
    Args:
        prompt: The text prompt to send
        model: The model to use
        cache_file: Path to the cache file
        options: Additional options for the API call
        
    Returns:
        The API response (either from cache or fresh)
    """
    if options is None:
        options = {}
    
    # Create cache key from prompt and model
    import hashlib
    cache_key = hashlib.md5((prompt + model).encode()).hexdigest()
    
    # Check if we should use caching
    if cache_file:
        cache = {}
        
        # Load existing cache if available
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except:
                logger.warning(f"Failed to load cache from {cache_file}, creating new cache")
        
        # Check if response is in cache
        if cache_key in cache:
            logger.info(f"Using cached response for prompt: {prompt[:50]}...")
            return cache[cache_key]
    
    # Call the API
    response = call_openai_api(prompt, model, options)
    
    # Cache the response if caching is enabled
    if cache_file:
        # Convert response to JSON for caching
        response_json = {
            "choices": [{"message": {"content": response.choices[0].message.content}}]
        }
        
        cache[cache_key] = response_json
        
        # Save the updated cache
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
        except:
            logger.warning(f"Failed to save cache to {cache_file}")
    
    return response

def convert_english_to_acep_with_ontology(
    text: str, 
    context: Dict[str, Any], 
    llm_options: Dict[str, Any],
    ontology_mapper: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert English text to ACEP representation using an LLM with ontology standardization.
    
    Args:
        text: Natural language text to convert
        context: Context information
        llm_options: Options for the LLM API call
        ontology_mapper: The ontology mapper
        
    Returns:
        The ACEP representation with standardized ontology terms
    """
    domain = context.get("domain", "general")
    certainty = context.get("certainty", 0.9)
    entity_id = context.get("entity_id", "")
    
    # First standardize the text using the ontology
    if "if" in text.lower() and "then" in text.lower():
        # It's a rule, standardize as rule
        standardized = standardize_rule_text(
            text, 
            ontology_mapper, 
            call_openai_api, 
            llm_options
        )
        
        # Check if we have ontology terms for both parts
        if (standardized["antecedent"]["term"] != "UNKNOWN" and 
            standardized["consequent"]["term"] != "UNKNOWN"):
            
            # Create an enhanced prompt with standardized terms
            prompt = f"""
            Convert this rule to ACEP (AI Conceptual Exchange Protocol) representation:
            
            Original text: "{text}"
            Domain: {domain}
            Entity: {entity_id}
            
            Standardized antecedent: {standardized["antecedent"]["term"]}
            Standardized consequent: {standardized["consequent"]["term"]}
            
            Use these standardized terms in your ACEP representation, with this format:
            
            ```json
            {{
              "identifier": "{standardized['antecedent']['term']}_implies_{standardized['consequent']['term']}",
              "type": "concept",
              "content": {{
                "concept": "conditional_relationship"
              }},
              "attributes": {{
                "antecedent": "{standardized['antecedent']['term']}",
                "consequent": "{standardized['consequent']['term']}",
                "certainty": {certainty},
                "original_text": "{text}"
              }}
            }}
            ```
            
            Return only the JSON with no additional text.
            """
        else:
            # Fallback to standard prompt without ontology terms
            prompt = f"""
            Convert this text to ACEP (AI Conceptual Exchange Protocol) representation:
            
            Text: "{text}"
            Domain: {domain}
            Entity: {entity_id}
            
            For conditional statements (if-then), return a JSON like:
            
            ```json
            {{
              "identifier": "consequent_if_antecedent",
              "type": "concept",
              "content": {{
                "concept": "conditional_relationship"
              }},
              "attributes": {{
                "antecedent": "condition",
                "consequent": "result",
                "certainty": {certainty}
              }}
            }}
            ```
            
            Return only the JSON with no additional text.
            """
    else:
        # It's a fact, standardize as fact
        standardized = standardize_fact_text(
            text, 
            ontology_mapper, 
            call_openai_api, 
            llm_options
        )
        
        # Check if we have an ontology term
        if standardized["term"] != "UNKNOWN":
            # Create an enhanced prompt with standardized term
            prompt = f"""
            Convert this fact to ACEP (AI Conceptual Exchange Protocol) representation:
            
            Original text: "{text}"
            Domain: {domain}
            Entity: {entity_id}
            
            Standardized term: {standardized["term"]}
            
            Use this standardized term in your ACEP representation, with this format:
            
            ```json
            {{
              "identifier": "{entity_id}_{standardized['term']}",
              "type": "concept",
              "content": {{
                "concept": "fact"
              }},
              "attributes": {{
                "term": "{standardized['term']}",
                "certainty": {certainty},
                "entity_id": "{entity_id}",
                "original_text": "{text}"
              }}
            }}
            ```
            
            Return only the JSON with no additional text.
            """
        else:
            # Fallback to standard prompt without ontology term
            prompt = f"""
            Convert this fact to ACEP (AI Conceptual Exchange Protocol) representation:
            
            Text: "{text}"
            Domain: {domain}
            Entity: {entity_id}
            
            Return a JSON like:
            
            ```json
            {{
              "identifier": "{entity_id}_fact",
              "type": "concept",
              "content": {{
                "concept": "fact"
              }},
              "attributes": {{
                "text": "{text}",
                "certainty": {certainty},
                "entity_id": "{entity_id}"
              }}
            }}
            ```
            
            Return only the JSON with no additional text.
            """
    
    try:
        # Call the LLM
        response = call_openai_api(prompt, llm_options.get("model", "gpt-4"), llm_options)
        acep_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON from the response
        import re
        
        # Look for JSON in the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', acep_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # If no JSON code block found, try to parse the entire response
            json_str = acep_text
        
        try:
            acep_data = json.loads(json_str)
            logger.info(f"Successfully parsed ACEP JSON for: {text[:30]}...")
            
            # Generate vector or use one from the standardized components
            vector_dimension = context.get("vector_dimension", 10000)
            
            from .vector_operations import generate_vector, bind_vectors
            
            # Generate vectors for components if available
            if (standardized.get("is_conditional", False) and 
                standardized["antecedent"]["term"] != "UNKNOWN" and 
                standardized["consequent"]["term"] != "UNKNOWN"):
                
                antecedent_vector = generate_vector(standardized["antecedent"]["term"], vector_dimension)
                consequent_vector = generate_vector(standardized["consequent"]["term"], vector_dimension)
                vector = bind_vectors(antecedent_vector, consequent_vector)
                
                # Add component vectors to ACEP representation
                acep_data["component_vectors"] = {
                    "antecedent": antecedent_vector,
                    "consequent": consequent_vector
                }
            else:
                # Generate a simple vector based on the identifier
                vector = generate_vector(acep_data.get("identifier", ""), vector_dimension)
            
            # Add vector to ACEP representation
            acep_data["vector"] = vector
            
            # Add standardization metadata
            acep_data["standardization"] = {
                "used_ontology": True,
                "ontology_domain": domain
            }
            
            if standardized.get("is_conditional", False):
                acep_data["standardization"]["antecedent_term"] = standardized["antecedent"]["term"]
                acep_data["standardization"]["antecedent_confidence"] = standardized["antecedent"]["confidence"]
                acep_data["standardization"]["consequent_term"] = standardized["consequent"]["term"]
                acep_data["standardization"]["consequent_confidence"] = standardized["consequent"]["confidence"]
            else:
                acep_data["standardization"]["term"] = standardized["term"]
                acep_data["standardization"]["confidence"] = standardized["confidence"]
            
            return acep_data
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ACEP JSON: {e}")
            logger.error(f"Raw text: {json_str}")
            # Fall back to a simple representation
            return {
                "identifier": f"{entity_id}_fallback_{hash(text) % 1000}",
                "type": "concept",
                "content": {"concept": "fact"},
                "attributes": {
                    "text": text,
                    "certainty": certainty,
                    "entity_id": entity_id
                },
                "vector": generate_vector(text, vector_dimension)
            }
            
    except Exception as e:
        logger.error(f"Error converting text to ACEP: {str(e)}")
        raise

def process_rules_with_ontology(
    rules: List[Dict[str, Any]],
    llm_options: Dict[str, Any],
    domain_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Process rules using ontology-based standardization.
    
    Args:
        rules: List of rule dictionaries from input configuration
        llm_options: Options for the LLM API
        domain_config: Domain configuration with ontology
        
    Returns:
        List of processed rules with standardized ontology terms
    """
    # Create ontology mapper
    ontology_mapper = create_ontology_mapper(domain_config)
    
    processed_rules = []
    
    for rule in rules:
        rule_text = rule.get("text", "")
        rule_certainty = rule.get("certainty", 0.9)
        
        if not rule_text:
            logger.warning(f"Skipping empty rule")
            continue
        
        try:
            # Convert to ACEP with ontology standardization
            context = {
                "domain": domain_config.get("domain", "general"),
                "certainty": rule_certainty,
                "vector_dimension": domain_config.get("vector_dimension", 10000)
            }
            
            acep_representation = convert_english_to_acep_with_ontology(
                rule_text,
                context,
                llm_options,
                ontology_mapper
            )
            
            # Ensure certainty from original rule is preserved
            acep_representation["attributes"]["certainty"] = rule_certainty
            
            # Add to processed rules
            processed_rules.append(acep_representation)
            
            logger.info(f"Processed rule with ontology: {acep_representation['identifier']}")
            
        except Exception as e:
            logger.error(f"Failed to process rule with ontology: {str(e)}")
    
    return processed_rules

def process_facts_with_ontology(
    facts: List[Dict[str, Any]],
    entity_id: str,
    llm_options: Dict[str, Any],
    domain_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Process facts using ontology-based standardization.
    
    Args:
        facts: List of fact dictionaries
        entity_id: ID of the entity
        llm_options: Options for the LLM API
        domain_config: Domain configuration with ontology
        
    Returns:
        List of processed facts with standardized ontology terms
    """
    # Create ontology mapper
    ontology_mapper = create_ontology_mapper(domain_config)
    
    processed_facts = []
    
    for fact in facts:
        fact_text = fact.get("text", "")
        fact_certainty = fact.get("certainty", 0.9)
        
        if not fact_text:
            logger.warning(f"Skipping empty fact")
            continue
        
        try:
            # Convert to ACEP with ontology standardization
            context = {
                "domain": domain_config.get("domain", "general"),
                "entity_id": entity_id,
                "certainty": fact_certainty,
                "vector_dimension": domain_config.get("vector_dimension", 10000)
            }
            
            acep_representation = convert_english_to_acep_with_ontology(
                fact_text,
                context,
                llm_options,
                ontology_mapper
            )
            
            # Ensure certainty from original fact is preserved
            acep_representation["attributes"]["certainty"] = fact_certainty
            
            # Ensure entity_id is set
            acep_representation["attributes"]["entity_id"] = entity_id
            
            # Add to processed facts
            processed_facts.append(acep_representation)
            
            logger.info(f"Processed fact with ontology: {acep_representation['identifier']}")
            
        except Exception as e:
            logger.error(f"Failed to process fact with ontology: {str(e)}")
    
    return processed_facts
