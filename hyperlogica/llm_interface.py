#!/usr/bin/env python3
"""
LLM Interface Module for Hyperlogica

This module provides functionality for interfacing with Language Models,
specifically for converting between natural language and ACEP representations,
with enhanced vector generation and processing.
"""

import os
import re
import json
import time
import logging
import hashlib
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

# Import vector operations
from .vector_operations import (
    generate_vector, normalize_vector, bind_vectors,
    unbind_vectors, bundle_vectors, calculate_similarity
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for API key
if "OPENAI_API_KEY" not in os.environ:
    logger.warning("OpenAI API key not found in environment variables.")

# Import backoff for retries - with proper error handling if not installed
try:
    import backoff
    has_backoff = True
except ImportError:
    logger.warning("backoff package not installed. No automatic retries will be performed.")
    has_backoff = False

# Import OpenAI with error handling
try:
    from openai import OpenAI
    has_openai = True
except ImportError:
    logger.warning("OpenAI package not installed. LLM functionality will be limited.")
    has_openai = False

# Define a fallback decorator if backoff is not available
if not has_backoff:
    def fallback_decorator(func):
        """Fallback decorator when backoff is not available"""
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    retry_decorator = fallback_decorator
else:
    # Use backoff for retries
    retry_decorator = backoff.on_exception(backoff.expo, Exception, max_tries=3)

# Export this function explicitly
@retry_decorator
def convert_english_to_acep(text: str, context: Dict[str, Any], llm_options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convert English text to ACEP representation using an LLM and generate compositional vector.
    Uses a controlled vocabulary from the ontology for consistent term usage.
    
    Args:
        text (str): English text to convert
        context (Dict[str, Any]): Context information including domain and certainty
        llm_options (Dict[str, Any]): Options for the LLM API call
        
    Returns:
        Dict[str, Any]: ACEP representation with vector
    """
    if not has_openai:
        raise ImportError("OpenAI package is required for this function")
    
    # Set default LLM options if not provided
    if llm_options is None:
        llm_options = {
            "model": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 2000
        }
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Set up context values
    domain = context.get("domain", "general")
    certainty = context.get("certainty", 0.9)
    entity_id = context.get("entity_id", "")
    vector_dim = context.get("vector_dimension", 10000)
    
    # Extract domain configuration with ontology from context
    domain_config = context.get("domain_config", {})
    finance_ontology = domain_config.get("finance_ontology", {})
    
    # Format the ontology for the prompt
    ontology_terms = []
    ontology_examples = []
    
    # Extract terms and provide examples for each category
    for category, terms_dict in finance_ontology.items():
        category_terms = []
        for term, phrases in terms_dict.items():
            ontology_terms.append(term)
            if phrases and len(phrases) > 0:
                category_terms.append(f"- {term}: {phrases[0]}")
        
        if category_terms:
            ontology_examples.append(f"{category.upper()} CATEGORY:")
            ontology_examples.extend(category_terms)
    
    # Format controlled vocab for the prompt
    formatted_terms = "\n".join([f"- {term}" for term in ontology_terms])
    formatted_examples = "\n".join(ontology_examples)
    
    # Create prompt for ACEP angle-bracket syntax with controlled vocabulary
    prompt = f"""
    Convert this text to ACEP (AI Conceptual Exchange Protocol) representation using angle-bracket syntax:
    
    Text: {text}
    Domain: {domain}
    Entity: {entity_id}
    
    IMPORTANT: When creating the ACEP representation, ONLY use terms from the following controlled vocabulary:
    
    {formatted_terms}
    
    Here are examples of what these terms represent:
    
    {formatted_examples}
    
    For conditional statements (if-then), use the format:
    <{{concept:ONTOLOGY_TERM_FOR_CONDITION}}> → <{{causal:{certainty}}}> → <{{concept:ONTOLOGY_TERM_FOR_RESULT}}>
    
    For facts or statements, use:
    <{{concept:ONTOLOGY_TERM, certainty:{certainty}}}> 
    
    Example for "If a company has low P/E ratio, then it is undervalued":
    <{{concept:PE_RATIO_LOW}}> → <{{causal:0.9}}> → <{{concept:STOCK_UNDERVALUED}}>
    
    Then parse this into structured JSON with the following format:
    
    {{
      "identifier": "unique_id_based_on_content",
      "type": "concept",
      "content": {{
        "concept": "conditional_relationship" or "fact"
      }},
      "attributes": {{
        "antecedent": "ONTOLOGY_TERM" (for conditional only),
        "consequent": "ONTOLOGY_TERM" (for conditional only),
        "certainty": {certainty},
        "entity_id": "{entity_id}",
        "domain": "{domain}",
        "ontology_term": "THE_MAIN_ONTOLOGY_TERM_THAT_APPLIES",
        "text": "{text}"
      }}
    }}
    """
    
    # Configure API call
    model = llm_options.get("model", "gpt-4")
    temperature = llm_options.get("temperature", 0.0)
    max_tokens = llm_options.get("max_tokens", 2000)
    
    logging.info(f"Converting to ACEP with controlled vocabulary: {text[:50]}...")
    
    try:
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
        
        # Extract the response text containing ACEP angle-bracket notation
        acep_text = response.choices[0].message.content.strip()
        
        # Log the full ACEP text
        logging.info(f"ACEP response for '{text[:30]}...': {acep_text}")
        
        # Try to parse JSON from the response
        import re
        import json
        
        # Look for JSON in the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', acep_text)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                acep_data = json.loads(json_str)
                logging.info(f"Successfully parsed JSON from ACEP response: {json.dumps(acep_data)}")
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON from ACEP response: {json_str}")
                acep_data = None
        else:
            logging.warning("No JSON found in ACEP response")
            acep_data = None
        
        # Generate an identifier based on the text
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        identifier = f"{entity_id}_{text_hash}" if entity_id else f"concept_{text_hash}"
        
        # Create a compositional vector based on the content
        vector_dimension = context.get("vector_dimension", 10000)
        
        # Check if this is a conditional (rule) or fact
        is_conditional_statement = "→" in acep_text or "->" in acep_text or "if" in text.lower() and "then" in text.lower()
        
        # Use the parsed JSON data if available
        if acep_data and isinstance(acep_data, dict):
            # Use the parsed structure
            identifier = acep_data.get("identifier", identifier)
            
            # Get the ontology term if available
            ontology_term = acep_data.get("attributes", {}).get("ontology_term", "")
            
            if is_conditional_statement:
                antecedent = acep_data.get("attributes", {}).get("antecedent", "")
                consequent = acep_data.get("attributes", {}).get("consequent", "")
                json_certainty = acep_data.get("attributes", {}).get("certainty", certainty)
                
                # Log the parsed components
                logging.info(f"Using parsed components - ID: {identifier}, Antecedent: {antecedent}, Consequent: {consequent}, Ontology: {ontology_term}")
                
                if antecedent and consequent:
                    # Generate component vectors
                    antecedent_vector = generate_vector(antecedent, dimension=vector_dimension)
                    consequent_vector = generate_vector(consequent, dimension=vector_dimension)
                    
                    # Bind the antecedent and consequent vectors to create rule vector
                    vector = bind_vectors(antecedent_vector, consequent_vector)
                    
                    # Create ACEP representation with rule components
                    acep_representation = {
                        "type": acep_data.get("type", "acep_relation"),
                        "identifier": identifier,
                        "acep_text": acep_text,
                        "original_text": text,
                        "attributes": {
                            "certainty": json_certainty,
                            "entity_id": entity_id,
                            "domain": domain,
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "conditional": True,
                            "rule_text": text,
                            "ontology_term": ontology_term
                        },
                        "component_vectors": {
                            "antecedent": antecedent_vector,
                            "consequent": consequent_vector
                        },
                        "vector": vector
                    }
                    logging.info(f"Created conditional representation with component vectors for {identifier}")
                    return acep_representation
            else:
                # For facts, use the ontology term if available
                if ontology_term:
                    vector = generate_vector(ontology_term, dimension=vector_dimension)
                    logging.info(f"Generated vector for ontology term: {ontology_term}")
                else:
                    # Extract keywords as fallback
                    keywords = extract_keywords(text)
                    if keywords:
                        logging.info(f"Extracted keywords for {identifier}: {', '.join(keywords)}")
                        # Generate a vector for each keyword and bundle them
                        keyword_vectors = [generate_vector(kw, dimension=vector_dimension) for kw in keywords]
                        vector = bundle_vectors(keyword_vectors)
                    else:
                        logging.warning(f"No ontology term or keywords found for {identifier}, using generic vector")
                        vector = generate_vector(identifier, dimension=vector_dimension)
                
                # Create fact representation
                acep_representation = {
                    "type": acep_data.get("type", "acep_concept"),
                    "identifier": identifier,
                    "acep_text": acep_text,
                    "original_text": text,
                    "attributes": {
                        "certainty": acep_data.get("attributes", {}).get("certainty", certainty),
                        "entity_id": entity_id,
                        "domain": domain,
                        "text": text,
                        "fact_text": text,
                        "ontology_term": ontology_term
                    },
                    "vector": vector
                }
                logging.info(f"Created fact representation with ontology term for {identifier}")
                return acep_representation
        
        # Fallback processing if JSON parsing failed
        if is_conditional_statement:
            # Extract antecedent and consequent
            try:
                parts = text.lower().split("then")
                antecedent = parts[0].replace("if", "", 1).strip()
                consequent = parts[1].strip().rstrip(".")
                
                # Generate component vectors
                antecedent_vector = generate_vector(antecedent, dimension=vector_dimension)
                consequent_vector = generate_vector(consequent, dimension=vector_dimension)
                
                # Bind the antecedent and consequent vectors to create rule vector
                vector = bind_vectors(antecedent_vector, consequent_vector)
                
                # Create ACEP representation with rule components
                acep_representation = {
                    "type": "acep_relation",
                    "identifier": identifier,
                    "acep_text": acep_text,
                    "original_text": text,
                    "attributes": {
                        "certainty": certainty,
                        "entity_id": entity_id,
                        "domain": domain,
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "conditional": True,
                        "rule_text": text
                    },
                    "component_vectors": {
                        "antecedent": antecedent_vector,
                        "consequent": consequent_vector
                    },
                    "vector": vector
                }
            except Exception as e:
                logging.warning(f"Error creating compositional vector for conditional: {e}")
                # Fallback to simple vector generation
                vector = generate_vector(identifier, dimension=vector_dimension)
                acep_representation = {
                    "type": "acep_relation",
                    "identifier": identifier,
                    "acep_text": acep_text,
                    "original_text": text,
                    "attributes": {
                        "certainty": certainty,
                        "entity_id": entity_id,
                        "domain": domain,
                        "conditional": True,
                        "rule_text": text
                    },
                    "vector": vector
                }
        else:
            # For facts, generate vectors based on key concepts in the text
            keywords = extract_keywords(text)
            if keywords:
                logging.info(f"Extracted keywords for {identifier}: {', '.join(keywords)}")
                # Generate a vector for each keyword and bundle them
                keyword_vectors = [generate_vector(kw, dimension=vector_dimension) for kw in keywords]
                vector = bundle_vectors(keyword_vectors)
            else:
                # Fallback if no keywords found
                logging.warning(f"No keywords found for {identifier}, using generic vector")
                vector = generate_vector(identifier, dimension=vector_dimension)
            
            acep_representation = {
                "type": "acep_concept",
                "identifier": identifier,
                "acep_text": acep_text,
                "original_text": text,
                "attributes": {
                    "certainty": certainty,
                    "entity_id": entity_id,
                    "domain": domain,
                    "text": text,
                    "fact_text": text
                },
                "vector": vector
            }
        
        logging.info(f"Successfully converted to ACEP: {identifier}")
        return acep_representation
        
    except Exception as e:
        logging.error(f"Error converting text to ACEP: {str(e)}")
        raise


def extract_keywords(text: str) -> List[str]:
    """Extract key concepts from text for vector composition."""
    # Simple implementation - extract nouns and key numbers
    import re
    
    # Extract potential keywords (nouns, numbers with units)
    keywords = []
    
    # Simple pattern for numbers with units
    number_pattern = r'\b\d+(?:\.\d+)?\s*(?:%|percent|ratio|times|months|quarters|years)\b'
    numbers = re.findall(number_pattern, text.lower())
    keywords.extend(numbers)
    
    # Add important financial terms if present
    financial_terms = [
        "pe ratio", "revenue growth", "profit margin", "debt-to-equity", 
        "return on equity", "price", "revenue", "margin", "debt", "equity", 
        "growth", "analyst", "rating", "undervalued", "overvalued"
    ]
    
    for term in financial_terms:
        if term in text.lower():
            keywords.append(term)
    
    # Return unique keywords
    return list(set(keywords))


@retry_decorator
def convert_acep_to_english(acep_representation: Dict[str, Any], context: Dict[str, Any], llm_options: Dict[str, Any] = None) -> str:
    """
    Convert ACEP representation to English text using LLM.
    
    Args:
        acep_representation (dict): Structured ACEP representation to convert.
        context (dict): Contextual information about the domain and entity.
        llm_options (dict, optional): Options for the LLM API call.
        
    Returns:
        str: Natural language English text representing the ACEP content.
    """
    if not has_openai:
        raise ImportError("OpenAI package is required for this function")
    
    # Set default LLM options if not provided
    if llm_options is None:
        llm_options = {
            "model": "gpt-4",
            "temperature": 0.3,
            "max_tokens": 1000
        }
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Get the identifier for logging
    identifier = acep_representation.get("identifier", "unknown")
    logger.info(f"Converting ACEP to English: {identifier}")
    
    # Extract the ACEP text if available, or use the original text
    acep_text = acep_representation.get("acep_text", "")
    original_text = acep_representation.get("original_text", "")
    
    # Create prompt
    prompt = f"""
    Convert this ACEP (AI Conceptual Exchange Protocol) representation into natural language:
    
    Original text: {original_text}
    ACEP representation: {acep_text}
    Domain: {context.get('domain', 'general')}
    Entity ID: {context.get('entity_id', '')}
    
    Generate clear, concise natural language that:
    - Accurately conveys the same meaning
    - Includes appropriate qualifiers to express certainty
    - Uses domain-appropriate terminology
    - Is suitable for human readers
    """
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model=llm_options.get("model", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert in translating AI knowledge representations into clear natural language."},
                {"role": "user", "content": prompt}
            ],
            temperature=llm_options.get("temperature", 0.3),
            max_tokens=llm_options.get("max_tokens", 1000)
        )
        
        # Extract the text
        english_text = response.choices[0].message.content.strip()
        logger.info(f"Successfully converted ACEP to English: {identifier}")
        
        return english_text
        
    except Exception as e:
        logger.error(f"Error converting ACEP to English: {str(e)}")
        # Return original text as fallback if conversion fails
        return original_text


@retry_decorator
def generate_explanation(reasoning_trace: Dict[str, Any], context: Dict[str, Any], llm_options: Dict[str, Any] = None) -> str:
    """
    Generate a natural language explanation from a reasoning trace.
    
    Args:
        reasoning_trace (dict): Reasoning trace data structure.
        context (dict): Contextual information including domain and recommendation.
        llm_options (dict, optional): Options for the LLM API call.
        
    Returns:
        str: Natural language explanation of the reasoning process.
    """
    if not has_openai:
        raise ImportError("OpenAI package is required for this function")
    
    logger.info("Generating explanation from reasoning trace")
    
    # Set default LLM options if not provided
    if llm_options is None:
        llm_options = {
            "model": "gpt-4",
            "temperature": 0.4,
            "max_tokens": 1500
        }
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Prepare the reasoning trace for the prompt
    # Remove large vectors to keep prompt size manageable
    def remove_vectors(obj):
        if isinstance(obj, dict):
            return {k: remove_vectors(v) for k, v in obj.items() if k != "vector"}
        elif isinstance(obj, list):
            return [remove_vectors(item) for item in obj]
        else:
            return obj
    
    clean_trace = remove_vectors(reasoning_trace)
    
    # Create the prompt
    prompt = f"""
    Explain the following reasoning process in clear, natural language:
    
    Reasoning trace:
    {json.dumps(clean_trace, indent=2)}
    
    Context:
    - Domain: {context.get('domain', 'general')}
    - Entity: {context.get('entity_id', '')}
    - Final recommendation: {context.get('recommendation', '')}
    - Confidence level: {context.get('certainty', 0.5):.2%}
    
    Your explanation should:
    1. Start with the final recommendation and its confidence level
    2. Explain the key factors that led to this conclusion
    3. Describe the logical steps in the reasoning process
    4. Use domain-appropriate terminology
    5. Be understandable to a non-technical audience
    
    Format the explanation as a well-structured paragraph that clearly explains the reasoning process
    behind the recommendation, highlighting the most important evidence and how certainty was determined.
    """
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model=llm_options.get("model", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert at explaining complex reasoning processes in clear, simple language."},
                {"role": "user", "content": prompt}
            ],
            temperature=llm_options.get("temperature", 0.4),
            max_tokens=llm_options.get("max_tokens", 1500)
        )
        
        # Extract the explanation
        explanation = response.choices[0].message.content.strip()
        logger.info("Successfully generated explanation from reasoning trace")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Failed to generate explanation: {str(e)}")
        return f"Unable to generate explanation due to an error: {str(e)}"


def create_embedding(text: str, model: str = "text-embedding-ada-002") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create an embedding vector for text using OpenAI's embedding API.
    
    Args:
        text (str): Text to embed
        model (str, optional): Embedding model to use. Defaults to "text-embedding-ada-002".
        
    Returns:
        tuple: (embedding_vector, metadata)
            - embedding_vector (np.ndarray): The embedding vector
            - metadata (dict): Information about the embedding
    """
    if not has_openai:
        raise ImportError("OpenAI package is required for this function")
    
    try:
        logger.info(f"Creating embedding for text: {text[:30]}...")
        
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Call the embedding API
        response = client.embeddings.create(
            model=model,
            input=text
        )
        
        # Extract the embedding
        embedding = response.data[0].embedding
        
        # Convert to numpy array
        embedding_vector = np.array(embedding)
        
        # Create metadata
        metadata = {
            "dimensions": len(embedding),
            "model": model,
            "text": text[:100] + "..." if len(text) > 100 else text
        }
        
        return embedding_vector, metadata
    
    except Exception as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        logger.warning("Falling back to deterministic vector generation")
        
        # Fallback to deterministic vector generation
        vector = generate_vector(text, 1536)  # 1536 is the dimension of ada embeddings
        
        metadata = {
            "dimensions": 1536,
            "model": "fallback-deterministic",
            "text": text[:100] + "..." if len(text) > 100 else text
        }
        
        return vector, metadata
    