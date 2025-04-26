#!/usr/bin/env python3
"""
LLM Interface Module for Hyperlogica

This module provides functionality for interfacing with Language Models,
specifically for converting between natural language and ACEP representations.
"""

import os
import re
import json
import time
import logging
import hashlib
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Check for API key
if "OPENAI_API_KEY" not in os.environ:
    logger.warning("OpenAI API key not found in environment variables.")


def create_english_to_acep_prompt(text: str, context: Dict[str, Any]) -> str:
    """
    Create a prompt for English to ACEP conversion.
    
    Args:
        text (str): The English text to be converted to ACEP.
        context (dict): Dictionary containing contextual information about the domain,
                        entity, and additional context useful for conversion.
        
    Returns:
        str: A formatted prompt string ready to be sent to the LLM API.
    """
    domain = context.get("domain", "general")
    entity_id = context.get("entity_id", "")
    certainty = context.get("certainty", 0.9)
    
    prompt = f"""
    Convert this statement to ACEP representation and return as JSON:

    Text: {text}

    For conditional statements (if-then), EXPLICITLY extract:
    1. The ANTECEDENT (the "if" part) as a separate field
    2. The CONSEQUENT (the "then" part) as a separate field
    3. Assign a meaningful identifier that reflects the content

    Example format for a conditional:
    {{
    "identifier": "pe_ratio_below_industry_implies_undervalued",
    "attributes": {{
        "rule_text": "If P/E ratio is below industry average, then the stock is potentially undervalued",
        "antecedent": "P/E ratio is below industry average",
        "consequent": "the stock is potentially undervalued",
        "conditional": true,
        "certainty": 0.8
    }}
    }}

    For facts, provide a meaningful identifier and appropriate attributes.
    """
    
    return prompt


def create_acep_to_english_prompt(acep_representation: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Create a prompt for ACEP to English conversion.
    
    Args:
        acep_representation (dict): ACEP structured representation to convert to English.
        context (dict): Dictionary containing contextual information about the domain,
                        entity, and additional context useful for conversion.
        
    Returns:
        str: A formatted prompt string ready to be sent to the LLM API.
    """
    domain = context.get("domain", "general")
    entity_id = context.get("entity_id", "")
    
    prompt = f"""
    Convert this structured AI representation into natural language:
    
    Structured representation:
    {json.dumps(acep_representation, indent=2)}
    
    Context:
    - Domain: {domain}
    - Entity ID: {entity_id}
    
    Generate clear, concise natural language that:
    - Accurately conveys the same meaning as the structured representation
    - Includes appropriate qualifiers to express certainty
    - Uses domain-appropriate terminology
    - Is suitable for human readers
    
    Format: A single paragraph of natural language text.
    """
    
    return prompt


def call_openai_api(prompt: str, model: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the OpenAI API with the given prompt, using built-in retry logic.
    Always returns JSON formatted responses.
    
    Args:
        prompt (str): The prompt to send to the API.
        model (str): Name of the OpenAI model to use (e.g., "gpt-4").
        options (dict): Additional API options such as temperature, max_tokens, etc.
        
    Returns:
        dict: The API response containing the model's output in JSON format.
        
    Raises:
        Exception: If the API call fails after retries.
    """
    try:
        logger.info(f"Calling OpenAI API with model: {model}")
        
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": options.get("temperature", 0.0),
            "max_tokens": options.get("max_tokens", 1000),
            "response_format": {"type": "json_object"}  # Always use json_object format
        }
        
        # Make the API call - OpenAI's client has built-in retry logic
        response = client.chat.completions.create(**api_params)
        return response
        
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise


def call_openai_api_cached(prompt: str, model: str, 
                          options: Dict[str, Any] = None, 
                          cache_file: str = "api_cache.json") -> Dict[str, Any]:
    """
    Call the OpenAI API with caching to reduce redundant API calls.
    
    Args:
        prompt (str): The prompt to send to the API.
        model (str): Name of the model to use.
        options (dict, optional): Additional API options. Defaults to None.
        cache_file (str, optional): File path for the cache. Defaults to "api_cache.json".
    
    Returns:
        dict: The API response, either from cache or fresh API call.
    """
    # Create a unique hash for this prompt+model+options combination
    options_str = json.dumps(options or {}, sort_keys=True)
    cache_key = hashlib.md5(f"{prompt}|{model}|{options_str}".encode()).hexdigest()
    
    # Try to load cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Error loading cache file {cache_file}, using empty cache")
    
    # Check if we have this result cached
    if cache_key in cache:
        logger.info("Using cached API response")
        return cache[cache_key]
    
    # If not in cache, make the actual API call
    response = call_openai_api(prompt, model, options or {})
    
    # Save to cache
    cache[cache_key] = response
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except IOError as e:
        logger.warning(f"Failed to write to cache file: {str(e)}")
    
    return response


def parse_acep_representation(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the LLM response into structured ACEP data.
    
    Args:
        response (dict): Raw response from the OpenAI API.
        
    Returns:
        dict: Parsed ACEP representation as a structured dictionary.
        
    Raises:
        ValueError: If the response cannot be parsed into a valid ACEP representation.
    """
    try:
        # Extract content from response
        content = response.choices[0].message.content
        
        # Log the raw content for debugging
        logger.debug(f"Raw LLM response content: {content}")
        
        # Handle potential JSON parsing errors
        try:
            # First, try to parse the content directly
            acep_data = json.loads(content)
        except json.JSONDecodeError as json_err:
            logger.warning(f"Initial JSON parse failed: {str(json_err)}")
            
            # Try to clean the content - sometimes the model outputs markdown-formatted JSON
            if "```json" in content:
                # Extract JSON from markdown code blocks
                json_block = content.split("```json")[1].split("```")[0].strip()
                acep_data = json.loads(json_block)
            else:
                # Try to find and extract a JSON object
                import re
                json_pattern = r'({.*})'
                match = re.search(json_pattern, content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    acep_data = json.loads(json_str)
                else:
                    raise ValueError(f"Could not extract valid JSON from response: {content[:100]}...")
        
        # Validate minimum required fields
        required_fields = ["identifier", "type"]
        for field in required_fields:
            if field not in acep_data:
                raise ValueError(f"Missing required field '{field}' in ACEP representation")
        
        # Ensure attributes dict exists
        if "attributes" not in acep_data:
            acep_data["attributes"] = {}
        
        # Ensure certainty exists in attributes
        if "certainty" not in acep_data["attributes"]:
            acep_data["attributes"]["certainty"] = 0.9  # Default high certainty
        
        return acep_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {str(e)}")
        raise ValueError(f"Invalid JSON in LLM response: {str(e)}")
    
    except (KeyError, IndexError, AttributeError) as e:
        logger.error(f"Invalid response structure: {str(e)}")
        raise ValueError(f"Invalid response structure: {str(e)}")


def get_english_to_acep_prompt(text: str, context: dict) -> str:
    """
    Generate a prompt for converting English text to ACEP representation.
    
    Args:
        text (str): The English text to convert to ACEP
        context (dict): Contextual information about the domain and entity
        
    Returns:
        str: A formatted prompt for the LLM
    """
    domain = context.get("domain", "general")
    certainty = context.get("certainty", 0.9)
    entity_id = context.get("entity_id", "")
    
    prompt = f"""
    Convert this statement to ACEP (AI Conceptual Exchange Protocol) representation:

    Text: {text}
    Domain: {domain}
    Entity: {entity_id}
    
    The ACEP representation must include these mandatory fields:
    1. "type" - Must be one of: "concept", "relation", or "operation"
    2. "identifier" - A unique, descriptive ID reflecting the content
    3. "attributes" - A dictionary of attributes including certainty
    
    For conditional statements (if-then), also include:
    - Set type to "relation"
    - "attributes.conditional" set to true
    - "attributes.antecedent" - The exact "if" part of the statement
    - "attributes.consequent" - The exact "then" part of the statement
    - "attributes.rule_text" - The full original text
    
    Example ACEP representation for a conditional rule:
    ```json
    {{
      "type": "relation",
      "identifier": "pe_ratio_below_industry_implies_undervalued",
      "attributes": {{
        "rule_text": "If P/E ratio is below industry average, then the stock is potentially undervalued",
        "antecedent": "P/E ratio is below industry average",
        "consequent": "the stock is potentially undervalued",
        "conditional": true,
        "certainty": 0.8
      }}
    }}
    ```
    
    Example ACEP representation for a fact:
    ```json
    {{
      "type": "concept",
      "identifier": "aapl_pe_ratio_below_industry",
      "attributes": {{
        "fact_text": "P/E ratio is 28.5, which is below the technology industry average of 32.8",
        "entity_id": "AAPL",
        "metric_type": "pe_ratio",
        "value": 28.5,
        "assessment": "below_average",
        "certainty": 0.95
      }}
    }}
    ```
    
    Ensure you include ALL mandatory fields and provide meaningful values that accurately represent the statement.
    Make the identifier descriptive and reflective of the content.
    
    Return only the ACEP representation as valid JSON.
    """
    
    return prompt


def convert_english_to_acep(text: str, context: dict, llm_options: dict) -> dict:
    """
    Convert English text to ACEP representation using an LLM.
    
    Args:
        text (str): English text to convert
        context (dict): Contextual information about the domain and entity
        llm_options (dict): Options for the LLM API call
        
    Returns:
        dict: ACEP representation of the input text
        
    Raises:
        ValueError: If the text cannot be converted to a valid ACEP representation
        OpenAIError: If the API call fails
    """

    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Generate the prompt using our helper function
    prompt = get_english_to_acep_prompt(text, context)
    
    # Log the conversion attempt
    logging.info(f"Converting to ACEP: {text[:50]}...")
    
    # Configure API call
    model = llm_options.get("model", "gpt-4-turbo")
    temperature = llm_options.get("temperature", 0.0)
    max_tokens = llm_options.get("max_tokens", 2000)
    
    logging.info(f"Calling OpenAI API with model: {model}")
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in AI knowledge representation."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        acep_representation = json.loads(content)
        
        # Validate the response has required fields
        required_fields = ["type", "identifier", "attributes"]
        for field in required_fields:
            if field not in acep_representation:
                raise ValueError(f"Missing required field '{field}' in ACEP representation")
                
        # Ensure certainty is set
        if "certainty" not in acep_representation.get("attributes", {}):
            acep_representation["attributes"]["certainty"] = context.get("certainty", 0.9)
            
        return acep_representation
        
    except Exception as e:
        logging.error(f"Error converting text to ACEP: {str(e)}")
        raise


def convert_acep_to_english(acep_representation: Dict[str, Any], context: Dict[str, Any], llm_options: Dict[str, Any] = None) -> str:
    """
    Convert ACEP representation to English text using LLM.
    
    Args:
        acep_representation (dict): Structured ACEP representation to convert.
        context (dict): Contextual information about the domain and entity.
        llm_options (dict, optional): Options for the LLM API call, 
                                     including model, temperature, etc.
        
    Returns:
        str: Natural language English text representing the ACEP content.
        
    Raises:
        ValueError: If the ACEP representation is invalid or cannot be converted.
    """
    logger.info(f"Converting from ACEP to English: {acep_representation['identifier']}")
    
    # Set default LLM options if not provided
    if llm_options is None:
        llm_options = {
            "temperature": 0.3,  # Slightly higher temperature for more natural language
            "max_tokens": 1000
        }
    
    model = llm_options.get("model", "gpt-4")
    
    # Create prompt for the LLM
    prompt = create_acep_to_english_prompt(acep_representation, context)
    
    # Call the API
    response = call_openai_api(prompt, model, llm_options)
    
    # Extract the text
    english_text = response.choices[0].message.content
    
    return english_text


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
    logger.info("Generating explanation from reasoning trace")
    
    # Set default LLM options if not provided
    if llm_options is None:
        llm_options = {
            "temperature": 0.4,  # Higher temperature for more creative explanations
            "max_tokens": 1500
        }
    
    model = llm_options.get("model", "gpt-4")
    
    # Create a prompt for explanation generation
    domain = context.get("domain", "general")
    entity_id = context.get("entity_id", "")
    recommendation = context.get("recommendation", "")
    certainty = context.get("certainty", 0.5)
    
    prompt = f"""
    Explain the following reasoning process in clear, natural language:
    
    Reasoning trace:
    {json.dumps(reasoning_trace, indent=2)}
    
    Context:
    - Domain: {domain}
    - Entity: {entity_id}
    - Final recommendation: {recommendation}
    - Confidence level: {certainty:.2%}
    
    Your explanation should:
    1. Start with the final recommendation and its confidence level
    2. Explain the key factors that led to this conclusion
    3. Describe the logical steps in the reasoning process
    4. Use domain-appropriate terminology
    5. Be understandable to a non-technical audience
    
    Format the explanation as a well-structured paragraph.
    """
    
    # We don't need to parse the response as JSON here, just return the text
    api_options = llm_options.copy()
    if 'response_format' in api_options:
        del api_options['response_format']  # Remove JSON response format
    
    try:
        response = call_openai_api(prompt, model, api_options)
        explanation = response.choices[0].message.content
        return explanation
    except Exception as e:
        logger.error(f"Failed to generate explanation: {str(e)}")
        return f"Unable to generate explanation due to an error: {str(e)}"


def create_embedding(text: str, model: str = "text-embedding-ada-002") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create an embedding vector for the given text using OpenAI's embedding API.
    
    Args:
        text (str): Text to embed.
        model (str, optional): Embedding model to use. Defaults to "text-embedding-ada-002".
    
    Returns:
        tuple: (embedding_vector, metadata)
            - embedding_vector (np.ndarray): The embedding vector.
            - metadata (dict): Information about the embedding (dimensions, model).
    """
    try:
        logger.info(f"Creating embedding for text: {text[:30]}...")
        
        # Call the embedding API - different endpoint from chat completions
        response = openai.embeddings.create(
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
        vector = generate_deterministic_vector(text, 1536)  # 1536 is the dimension of ada embeddings
        
        metadata = {
            "dimensions": 1536,
            "model": "fallback-deterministic",
            "text": text[:100] + "..." if len(text) > 100 else text
        }
        
        return vector, metadata


def generate_deterministic_vector(text: str, dimension: int = 10000) -> np.ndarray:
    """
    Generate a deterministic vector from text as a fallback when API is unavailable.
    
    Args:
        text (str): Input text to convert to a vector.
        dimension (int, optional): Dimensionality of the vector. Defaults to 10000.
    
    Returns:
        np.ndarray: A unit vector derived from the text in a deterministic way.
    """
    # Create a hash of the text to use as a seed
    text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    
    # Set the random seed for reproducibility
    np.random.seed(text_hash)
    
    # Generate a random vector
    vector = np.random.normal(0, 1, dimension)
    
    # Normalize to unit length
    vector = vector / np.linalg.norm(vector)
    
    return vector


def create_normalized_identifier(text: str, max_length: int = 50) -> str:
    """
    Create a normalized identifier from text.
    
    Args:
        text (str): Text to normalize.
        max_length (int, optional): Maximum length of the identifier. Defaults to 50.
    
    Returns:
        str: Normalized identifier suitable for use as a key.
    """
    # Convert to lowercase
    identifier = text.lower()
    
    # Replace non-alphanumeric characters with underscores
    import re
    identifier = re.sub(r'[^\w\s]', '', identifier)
    
    # Replace whitespace with underscores
    identifier = re.sub(r'\s+', '_', identifier)
    
    # Truncate if too long
    if len(identifier) > max_length:
        identifier = identifier[:max_length]
    
    return identifier


def extract_certainty_language(text: str) -> float:
    """
    Extract the certainty level from natural language expressions.
    
    Args:
        text (str): Text containing certainty expressions.
    
    Returns:
        float: Estimated certainty value between 0 and 1.
    """
    text = text.lower()
    
    # Define certainty mappings
    high_certainty = [
        "certainly", "definitely", "absolutely", "undoubtedly", "will",
        "is certain", "is guaranteed", "always", "undeniable"
    ]
    
    medium_high_certainty = [
        "very likely", "highly probable", "strongly suggest", "most likely",
        "almost certainly", "highly confident", "will probably"
    ]
    
    medium_certainty = [
        "likely", "probably", "suggests", "indicates", "tends to",
        "appears to", "seem", "often", "usually"
    ]
    
    medium_low_certainty = [
        "may", "might", "possibly", "perhaps", "could",
        "sometimes", "somewhat", "can", "potential"
    ]
    
    low_certainty = [
        "unlikely", "doubtful", "rarely", "seldom", "slight chance",
        "questionable", "improbable", "not likely"
    ]
    
    very_low_certainty = [
        "very unlikely", "highly doubtful", "almost certainly not",
        "very improbable", "virtually impossible", "remote chance"
    ]
    
    # Check for negations
    negations = ["not", "never", "no", "don't", "doesn't", "isn't", "aren't", "won't"]
    has_negation = any(neg in text.split() for neg in negations)
    
    # Determine certainty value
    if any(term in text for term in high_certainty) and not has_negation:
        return 0.95
    elif any(term in text for term in medium_high_certainty) and not has_negation:
        return 0.85
    elif any(term in text for term in medium_certainty) and not has_negation:
        return 0.7
    elif any(term in text for term in medium_low_certainty) and not has_negation:
        return 0.5
    elif any(term in text for term in low_certainty) and not has_negation:
        return 0.3
    elif any(term in text for term in very_low_certainty) and not has_negation:
        return 0.1
    elif any(term in text for term in high_certainty) and has_negation:
        return 0.05
    elif any(term in text for term in medium_high_certainty) and has_negation:
        return 0.15
    elif any(term in text for term in medium_certainty) and has_negation:
        return 0.3
    
    # Default value for text with no clear certainty indicators
    return 0.5


def is_conditional_statement(text: str) -> bool:
    """
    Detect if a statement is conditional (if-then structure).
    
    Args:
        text (str): Text to analyze.
    
    Returns:
        bool: True if the statement is conditional, False otherwise.
    """
    text = text.lower()
    
    # Common conditional indicators
    conditional_indicators = [
        "if", "when", "whenever", "unless", "provided that", "assuming that",
        "in case", "should", "as long as", "only if", "given that"
    ]
    
    # Common consequent indicators
    consequent_indicators = [
        "then", "will", "would", "should", "may", "might", "can", "could"
    ]
    
    # Check for conditional structure
    has_condition = any(indicator in text.split() for indicator in conditional_indicators)
    has_consequent = any(indicator in text.split() for indicator in consequent_indicators)
    
    return has_condition and has_consequent


def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extract a numeric value from text if present.
    
    Args:
        text (str): Text to analyze.
    
    Returns:
        float or None: Extracted numeric value, or None if no value found.
    """
    import re
    
    # Look for percentage values first
    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', text)
    if percentage_match:
        return float(percentage_match.group(1)) / 100
    
    # Look for regular numeric values
    numeric_match = re.search(r'(\d+(?:\.\d+)?)', text)
    if numeric_match:
        return float(numeric_match.group(1))
    
    return None


def extract_temporal_reference(text: str) -> Optional[str]:
    """
    Extract temporal references from text.
    
    Args:
        text (str): Text to analyze.
    
    Returns:
        str or None: Extracted temporal reference, or None if not found.
    """
    text = text.lower()
    
    # Common time references
    time_references = {
        "immediate": ["today", "now", "immediately", "current", "presently"],
        "short_term": ["tomorrow", "this week", "next week", "soon", "shortly", "days"],
        "medium_term": ["this month", "next month", "quarterly", "this quarter", "months"],
        "long_term": ["this year", "next year", "annual", "long-term", "years"],
        "past": ["yesterday", "last week", "last month", "last year", "previously"]
    }
    
    # Check for ISO dates (YYYY-MM-DD)
    import re
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if date_match:
        return date_match.group(0)
    
    # Check for time references
    for period, references in time_references.items():
        if any(ref in text for ref in references):
            return period
    
    return None


if __name__ == "__main__":
    # Simple test function to verify functionality
    def test_api_call():
        if "OPENAI_API_KEY" not in os.environ:
            return {
                "success": False,
                "error": "OpenAI API key not found in environment variables."
            }
        #openai.api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo", # Equivalent to GPT-4.1
                messages=[
                    {
                        "role": "user",
                        "content": "What are the advantages of vector-based AI communication? Respond in JSON format with keys 'summary' and 'points' (a list)."
                    },
                ],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"} # Changed from string to object with 'type' field
            )
            return {
                "success": True,
                "response": response.choices[0].message.content
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    print(test_api_call())