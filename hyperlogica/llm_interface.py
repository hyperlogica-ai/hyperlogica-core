# llm_interface.py
"""
LLM Interface for Hyperlogica System

This module provides functions for interfacing with Large Language Models (LLMs)
to convert between natural language and ACEP (AI Conceptual Exchange Protocol)
representations. It handles prompt creation, API calls, response parsing, and
utility functions for working with language.

The module follows a functional programming approach with no classes, just pure
functions designed to be composed together.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import openai
import numpy as np
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Initialize OpenAI client if API key is available
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    client = openai.OpenAI(api_key=api_key)
else:
    logging.warning("OpenAI API key not found in environment variables. Some functions may not work.")

def create_english_to_acep_prompt(text: str, context: Dict[str, Any]) -> str:
    """
    Create a prompt for English to ACEP conversion.
    
    Args:
        text (str): The English text to be converted to ACEP.
        context (dict): Dictionary containing contextual information about the domain,
                        entity, and additional context useful for conversion.
        
    Returns:
        str: A formatted prompt string ready to be sent to the LLM API for 
             converting the input text to ACEP representation.
    """
    domain = context.get("domain", "general")
    entity_id = context.get("entity_id", "")
    additional_context = context.get("additional_context", "")
    
    prompt = f"""
    You are an expert in converting natural language statements into a structured format called ACEP (AI Conceptual Exchange Protocol), which uses vector representations for precise AI-to-AI communication.

    Please convert the following text into ACEP format:

    [TEXT: {text}]

    If this is a rule or conditional statement:
    1. Identify the antecedent (if condition) and consequent (then result)
    2. Assign appropriate certainty values (0.0-1.0)
    3. Identify any domain-specific attributes (temporal references, valence, etc.)

    If this is a factual statement:
    1. Identify the core concept being described
    2. Extract any numeric values or assessments
    3. Assign appropriate certainty values (0.0-1.0)
    4. Identify any domain-specific attributes

    Context information:
    [DOMAIN: {domain}]
    [ENTITY: {entity_id}]
    [ADDITIONAL_CONTEXT: {additional_context}]

    Produce a complete ACEP representation with:
    - Concept identifier
    - Relationship markers (if applicable)
    - Attributes including certainty
    - Explanation of how you derived this representation

    Format your response as valid JSON.
    """
    
    return prompt.strip()

def create_acep_to_english_prompt(acep_representation: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Create a prompt for ACEP to English conversion.
    
    Args:
        acep_representation (dict): ACEP structured representation to convert to English.
        context (dict): Dictionary containing contextual information about the domain,
                        entity, and additional context useful for conversion.
        
    Returns:
        str: A formatted prompt string ready to be sent to the LLM API for
             converting the ACEP representation to natural English text.
    """
    domain = context.get("domain", "general")
    entity_id = context.get("entity_id", "")
    additional_context = context.get("additional_context", "")
    
    # Convert ACEP representation to JSON string for prompt
    acep_json = json.dumps(acep_representation, indent=2)
    
    prompt = f"""
    You are an expert in converting structured ACEP (AI Conceptual Exchange Protocol) representations into natural language that is precise, clear, and understandable to humans.

    Please convert the following ACEP representation into natural language:

    [ACEP_REPRESENTATION:
    {acep_json}
    ]

    Context information:
    [DOMAIN: {domain}]
    [ENTITY: {entity_id}]
    [ADDITIONAL_CONTEXT: {additional_context}]

    For concepts, provide a clear statement of the concept.
    For relationships, explain the relationship between concepts.
    For reasoning chains, explain the logical flow from premises to conclusion.

    Include:
    1. The main statement in clear, concise language
    2. The certainty level expressed in natural terms (e.g., "highly likely", "somewhat uncertain")
    3. Any relevant context from the ACEP attributes

    Avoid using technical terminology related to ACEP or vector representations.
    Format your response as a well-structured paragraph suitable for a business report.
    """
    
    return prompt.strip()

def create_reasoning_explanation_prompt(reasoning_trace: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Create a prompt for explaining reasoning traces in natural language.
    
    Args:
        reasoning_trace (dict): The reasoning trace to explain, containing steps, conclusions, etc.
        context (dict): Contextual information including domain, entity, recommendation, and certainty.
        
    Returns:
        str: A formatted prompt string ready to be sent to the LLM API for
             generating a natural language explanation of the reasoning process.
    """
    domain = context.get("domain", "general")
    entity_id = context.get("entity_id", "")
    recommendation = context.get("recommendation", "")
    certainty = context.get("certainty", 0.0)
    
    # Convert reasoning trace to JSON string for prompt
    trace_json = json.dumps(reasoning_trace, indent=2)
    
    prompt = f"""
    You are an expert in explaining AI reasoning processes in clear, understandable language for non-technical users.

    Please explain the following reasoning chain:

    [REASONING_TRACE:
    {trace_json}
    ]

    Context information:
    [DOMAIN: {domain}]
    [ENTITY: {entity_id}]
    [RECOMMENDATION: {recommendation}]
    [CERTAINTY: {certainty}]

    Your explanation should:
    1. Start with the final recommendation and its confidence level
    2. Outline the key factors that led to this conclusion
    3. Explain any particularly important reasoning steps
    4. Mention the balance of evidence (e.g., positive vs. negative signals)
    5. Use domain-appropriate language and examples

    Avoid technical jargon related to AI, vectors, or reasoning patterns.
    Focus on making the explanation accessible and convincing to a business audience.
    Format your response as a well-structured set of paragraphs with appropriate headings.
    """
    
    return prompt.strip()

def call_openai_api(prompt: str, model: str = "gpt-4", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Call the OpenAI API with the given prompt.
    
    Args:
        prompt (str): The prompt to send to the API.
        model (str): Name of the OpenAI model to use (e.g., "gpt-4").
        options (dict, optional): Additional API options such as temperature, max_tokens, etc.
        
    Returns:
        dict: The API response containing the model's output.
        
    Raises:
        openai.OpenAIError: If the API call fails due to authentication, rate limiting, or other errors.
        TimeoutError: If the API call times out.
    """
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
    if options is None:
        options = {}
    
    # Default options that can be overridden
    default_options = {
        "temperature": 0.0,
        "max_tokens": 2000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    # Merge default options with provided options
    api_options = {**default_options, **options}
    
    # Extract response_format if provided
    response_format = api_options.pop("response_format", {"type": "text"})
    
    # Set up retry mechanism
    max_retries = api_options.pop("max_retries", 3)
    retry_delay = api_options.pop("retry_delay", 2)
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            logging.info(f"Calling OpenAI API with model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                **api_options
            )
            
            elapsed_time = time.time() - start_time
            logging.info(f"API call completed in {elapsed_time:.2f} seconds")
            
            # Convert to dictionary for consistency
            return {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content,
                            "role": response.choices[0].message.role
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                "model": response.model,
                "object": response.object,
                "id": response.id,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "created": response.created
            }
            
        except (openai.RateLimitError, openai.APIConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"API error: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            raise
            
def parse_vector_representation(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the LLM response into structured data.
    
    Args:
        response (dict): Raw response from the OpenAI API.
        
    Returns:
        dict: Parsed ACEP representation as a structured dictionary.
        
    Raises:
        ValueError: If the response cannot be parsed into a valid ACEP representation.
        json.JSONDecodeError: If the response content is not valid JSON.
    """
    try:
        # Extract content from the response
        content = response["choices"][0]["message"]["content"]
        
        # Some LLMs might wrap JSON in markdown code blocks, so we attempt to extract it
        if "```json" in content and "```" in content:
            start_idx = content.find("```json") + 7
            end_idx = content.rfind("```")
            content = content[start_idx:end_idx].strip()
        elif "```" in content:
            # Handle code blocks without language specification
            start_idx = content.find("```") + 3
            end_idx = content.rfind("```")
            content = content[start_idx:end_idx].strip()
            
        # Parse the JSON content
        parsed_data = json.loads(content)
        
        # Validate required fields for ACEP representation
        required_fields = ["identifier", "type"]
        if not all(field in parsed_data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in parsed_data]
            raise ValueError(f"Missing required fields in ACEP representation: {missing_fields}")
            
        return parsed_data
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from response: {e}")
        logging.debug(f"Response content: {response['choices'][0]['message']['content']}")
        raise ValueError(f"Response is not valid JSON: {str(e)}")
    except KeyError as e:
        logging.error(f"Missing expected key in response: {e}")
        raise ValueError(f"Invalid response structure: {str(e)}")
    except Exception as e:
        logging.error(f"Error parsing vector representation: {e}")
        raise

def convert_english_to_acep(text: str, context: Dict[str, Any], llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert English text to ACEP representation using LLM.
    
    Args:
        text (str): English text to convert.
        context (dict): Contextual information about the domain and entity.
        llm_options (dict, optional): Options for the LLM API call, including model, temperature, etc.
        
    Returns:
        dict: Structured ACEP representation of the input text.
        
    Raises:
        ValueError: If the text cannot be converted to a valid ACEP representation.
        openai.OpenAIError: If the API call fails.
    """
    if llm_options is None:
        llm_options = {}
    
    # Set default model if not provided
    model = llm_options.get("model", "gpt-4")
    
    # Add response format for JSON
    llm_options["response_format"] = {"type": "json_object"}
    
    # Create the prompt
    prompt = create_english_to_acep_prompt(text, context)
    
    # Make the API call
    logging.info(f"Converting to ACEP: {text[:50]}{'...' if len(text) > 50 else ''}")
    response = call_openai_api(prompt, model, llm_options)
    
    # Parse the response
    try:
        acep_representation = parse_vector_representation(response)
        
        # Add metadata about the source text
        if "metadata" not in acep_representation:
            acep_representation["metadata"] = {}
        acep_representation["metadata"]["source_text"] = text
        acep_representation["metadata"]["context"] = context
        
        logging.info(f"Successfully converted to ACEP with identifier: {acep_representation.get('identifier', 'unknown')}")
        return acep_representation
        
    except Exception as e:
        logging.error(f"Failed to convert text to ACEP: {str(e)}")
        raise ValueError(f"Failed to convert text to ACEP representation: {str(e)}")

def convert_acep_to_english(acep_representation: Dict[str, Any], context: Dict[str, Any], llm_options: Optional[Dict[str, Any]] = None) -> str:
    """
    Convert ACEP representation to English text using LLM.
    
    Args:
        acep_representation (dict): Structured ACEP representation to convert.
        context (dict): Contextual information about the domain and entity.
        llm_options (dict, optional): Options for the LLM API call, including model, temperature, etc.
        
    Returns:
        str: Natural language English text representing the ACEP content.
        
    Raises:
        ValueError: If the ACEP representation is invalid or cannot be converted.
        openai.OpenAIError: If the API call fails.
    """
    if llm_options is None:
        llm_options = {}
    
    # Set default model if not provided
    model = llm_options.get("model", "gpt-4")
    
    # Create the prompt
    prompt = create_acep_to_english_prompt(acep_representation, context)
    
    # Make the API call
    logging.info(f"Converting ACEP to English: {acep_representation.get('identifier', 'unknown')}")
    response = call_openai_api(prompt, model, llm_options)
    
    # Extract the English text from the response
    english_text = response["choices"][0]["message"]["content"]
    
    logging.info(f"Successfully converted ACEP to English text")
    return english_text

def generate_explanation(reasoning_trace: Dict[str, Any], context: Dict[str, Any], llm_options: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a natural language explanation of a reasoning trace.
    
    Args:
        reasoning_trace (dict): The reasoning trace containing steps, conclusions, etc.
        context (dict): Context information including domain, entity, recommendation, etc.
        llm_options (dict, optional): Options for the LLM API call, including model, temperature, etc.
        
    Returns:
        str: Natural language explanation of the reasoning process.
        
    Raises:
        ValueError: If the reasoning trace is invalid or cannot be explained.
        openai.OpenAIError: If the API call fails.
    """
    if llm_options is None:
        llm_options = {}
    
    # Set default model if not provided
    model = llm_options.get("model", "gpt-4")
    
    # Create the prompt
    prompt = create_reasoning_explanation_prompt(reasoning_trace, context)
    
    # Make the API call
    logging.info(f"Generating explanation for reasoning trace with {len(reasoning_trace.get('steps', []))} steps")
    response = call_openai_api(prompt, model, llm_options)
    
    # Extract the explanation from the response
    explanation = response["choices"][0]["message"]["content"]
    
    logging.info(f"Successfully generated explanation")
    return explanation

def is_conditional_statement(text: str) -> bool:
    """
    Simple heuristic to determine if a text contains a conditional statement.
    
    Args:
        text (str): The text to analyze.
        
    Returns:
        bool: True if the text appears to contain a conditional statement, False otherwise.
    """
    lower_text = text.lower()
    
    # Check for common conditional constructs
    if_then_patterns = [
        "if" in lower_text and "then" in lower_text,
        "when" in lower_text and "then" in lower_text,
        "whenever" in lower_text and "then" in lower_text,
        "in case" in lower_text and "then" in lower_text
    ]
    
    return any(if_then_patterns)

def extract_certainty_language(text: str) -> float:
    """
    Extract a numerical certainty value from natural language expressions.
    
    Args:
        text (str): Text containing certainty expressions.
        
    Returns:
        float: Estimated certainty value between 0.0 and 1.0.
    """
    lower_text = text.lower()
    
    # Definite certainty expressions
    if any(phrase in lower_text for phrase in ["certainly", "definitely", "always", "absolutely", "guaranteed"]):
        return 0.95
    
    # High certainty expressions
    if any(phrase in lower_text for phrase in ["very likely", "highly probable", "strong chance", "usually"]):
        return 0.8
    
    # Moderate certainty expressions
    if any(phrase in lower_text for phrase in ["likely", "probably", "often", "should"]):
        return 0.7
    
    # Uncertain expressions
    if any(phrase in lower_text for phrase in ["possibly", "might", "may", "can", "sometimes"]):
        return 0.5
    
    # Low certainty expressions
    if any(phrase in lower_text for phrase in ["unlikely", "rarely", "seldom", "doubtful"]):
        return 0.3
    
    # Very low certainty expressions
    if any(phrase in lower_text for phrase in ["very unlikely", "highly doubtful", "almost never"]):
        return 0.1
    
    # Default moderate certainty if no expressions found
    return 0.7

def create_embedding(text: str, model: str = "text-embedding-ada-002") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a vector embedding for a text using OpenAI's embedding API.
    
    Args:
        text (str): Text to convert to a vector embedding.
        model (str, optional): Name of the embedding model to use. Defaults to "text-embedding-ada-002".
        
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: A tuple containing:
            - The embedding vector as a numpy array
            - Metadata about the embedding including model, dimensions, and usage
            
    Raises:
        ValueError: If the API key is not set or the embedding creation fails.
        openai.OpenAIError: If the API call fails.
    """
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    try:
        logging.info(f"Creating embedding for text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        start_time = time.time()
        response = client.embeddings.create(
            model=model,
            input=text
        )
        elapsed_time = time.time() - start_time
        
        # Extract the embedding
        embedding = np.array(response.data[0].embedding)
        
        metadata = {
            "model": response.model,
            "dimensions": len(embedding),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "elapsed_time": elapsed_time
        }
        
        logging.info(f"Created {metadata['dimensions']}-dimensional embedding in {elapsed_time:.2f} seconds")
        
        return embedding, metadata
        
    except Exception as e:
        logging.error(f"Failed to create embedding: {str(e)}")
        raise ValueError(f"Failed to create embedding: {str(e)}")

def generate_deterministic_vector(text: str, dimension: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a deterministic vector from text for when API embeddings aren't available.
    
    Args:
        text (str): Text to convert to a vector.
        dimension (int): Dimension of the vector to generate.
        seed (int, optional): Random seed for reproducibility. If None, will use hash of text.
        
    Returns:
        np.ndarray: A normalized vector of specified dimension.
    """
    # If no seed provided, create one from the text
    if seed is None:
        seed = hash(text) % (2**32)
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate a random vector
    vector = np.random.normal(0, 1, dimension)
    
    # Normalize to unit length
    vector = vector / np.linalg.norm(vector)
    
    return vector

def cache_api_response(prompt: str, response: Dict[str, Any], cache_file: str = "api_cache.json") -> None:
    """
    Cache an API response to a file to avoid redundant API calls.
    
    Args:
        prompt (str): The prompt that was sent to the API.
        response (Dict[str, Any]): The response received from the API.
        cache_file (str, optional): Path to the cache file. Defaults to "api_cache.json".
    """
    try:
        # Generate a key for the cache using a hash of the prompt
        cache_key = str(hash(prompt))
        
        # Load existing cache if it exists
        cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        
        # Add the new response to the cache
        cache[cache_key] = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        }
        
        # Save the updated cache
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
            
        logging.info(f"Cached API response for prompt hash {cache_key}")
        
    except Exception as e:
        logging.warning(f"Failed to cache API response: {str(e)}")

def get_cached_response(prompt: str, max_age: Optional[float] = None, cache_file: str = "api_cache.json") -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached API response if available and not expired.
    
    Args:
        prompt (str): The prompt to look up in the cache.
        max_age (float, optional): Maximum age of cached response in seconds. 
                                 If None, no expiration. Defaults to None.
        cache_file (str, optional): Path to the cache file. Defaults to "api_cache.json".
        
    Returns:
        Optional[Dict[str, Any]]: The cached response if found and not expired, None otherwise.
    """
    try:
        # Generate a key for the cache lookup
        cache_key = str(hash(prompt))
        
        # Check if cache file exists
        if not os.path.exists(cache_file):
            return None
        
        # Load the cache
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        
        # Check if the key exists in the cache
        if cache_key not in cache:
            return None
        
        # Check if the cached response is expired
        cached_entry = cache[cache_key]
        if max_age is not None:
            age = time.time() - cached_entry["timestamp"]
            if age > max_age:
                logging.info(f"Cached response expired (age: {age:.2f}s, max_age: {max_age:.2f}s)")
                return None
        
        logging.info(f"Retrieved cached API response for prompt hash {cache_key}")
        return cached_entry["response"]
        
    except Exception as e:
        logging.warning(f"Failed to retrieve cached response: {str(e)}")
        return None

def call_openai_api_cached(prompt: str, model: str = "gpt-4", options: Optional[Dict[str, Any]] = None, 
                         max_age: Optional[float] = None, cache_file: str = "api_cache.json") -> Dict[str, Any]:
    """
    Call the OpenAI API with caching to avoid redundant API calls.
    
    Args:
        prompt (str): The prompt to send to the API.
        model (str, optional): Name of the OpenAI model to use. Defaults to "gpt-4".
        options (Dict[str, Any], optional): Additional API options. Defaults to None.
        max_age (float, optional): Maximum age of cached response in seconds.
                                 If None, no expiration. Defaults to None.
        cache_file (str, optional): Path to the cache file. Defaults to "api_cache.json".
        
    Returns:
        Dict[str, Any]: The API response, either from cache or from a new API call.
        
    Raises:
        The same exceptions as call_openai_api.
    """
    # Check if response is in cache
    cached_response = get_cached_response(prompt, max_age, cache_file)
    if cached_response is not None:
        return cached_response
    
    # If not in cache or expired, make a new API call
    response = call_openai_api(prompt, model, options)
    
    # Cache the response
    cache_api_response(prompt, response, cache_file)
    
    return response

def create_normalized_identifier(text: str, max_length: int = 50) -> str:
    """
    Create a normalized identifier from text, suitable for use as an ACEP identifier.
    
    Args:
        text (str): Text to convert to an identifier.
        max_length (int, optional): Maximum length of the identifier. Defaults to 50.
        
    Returns:
        str: A normalized identifier containing only lowercase letters, numbers, and underscores.
    """
    # Convert to lowercase
    identifier = text.lower()
    
    # Remove punctuation and special characters
    identifier = re.sub(r'[^\w\s]', '', identifier)
    
    # Replace spaces with underscores
    identifier = re.sub(r'\s+', '_', identifier)
    
    # Remove any consecutive underscores
    identifier = re.sub(r'_+', '_', identifier)
    
    # Truncate if too long
    if len(identifier) > max_length:
        identifier = identifier[:max_length]
    
    # Remove trailing underscores
    identifier = identifier.rstrip('_')
    
    return identifier

def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extract a numeric value from text.
    
    Args:
        text (str): Text containing a numeric value.
        
    Returns:
        Optional[float]: The extracted numeric value, or None if no value is found.
    """
    # Look for percentages first
    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', text)
    if percentage_match:
        return float(percentage_match.group(1)) / 100
    
    # Look for general numbers
    number_match = re.search(r'(\d+(?:\.\d+)?)', text)
    if number_match:
        return float(number_match.group(1))
    
    return None

def extract_temporal_reference(text: str) -> Optional[str]:
    """
    Extract temporal references from text.
    
    Args:
        text (str): Text containing temporal references.
        
    Returns:
        Optional[str]: Extracted temporal reference, or None if no reference is found.
    """
    text = text.lower()
    
    # Check for specific time periods
    time_periods = [
        "today", "tomorrow", "yesterday", 
        "next week", "last week", "this week",
        "next month", "last month", "this month",
        "next year", "last year", "this year",
        "current quarter", "next quarter", "previous quarter",
        "short term", "long term", "medium term"
    ]
    
    for period in time_periods:
        if period in text:
            return period
    
    # Check for dates in format YYYY-MM-DD
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if date_match:
        return date_match.group(0)
    
    # Check for relative time expressions
    if re.search(r'in \d+ days?', text):
        return re.search(r'in \d+ days?', text).group(0)
    if re.search(r'in \d+ weeks?', text):
        return re.search(r'in \d+ weeks?', text).group(0)
    if re.search(r'in \d+ months?', text):
        return re.search(r'in \d+ months?', text).group(0)
    if re.search(r'in \d+ years?', text):
        return re.search(r'in \d+ years?', text).group(0)
    
    return None

# Example usage function
def example_usage():
    """
    Demonstrate how to use the LLM interface with a simple example.
    """
    # Set up OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Example text to convert
    text = "If a company's P/E ratio is below the industry average, then the stock might be undervalued."

    # Set up context for the conversion
    context = {
        "domain": "finance",
        "entity_id": "example_stock",
        "additional_context": "Stock valuation analysis"
    }
    
    try:
        print(f"Input text: {text}")
        
        # Convert the text to ACEP representation
        print("\nConverting text to ACEP representation...")
        acep_representation = convert_english_to_acep(text, context)
        print("\nACEP representation:")
        print(json.dumps(acep_representation, indent=2))
        
        # Convert the ACEP representation back to English
        print("\nConverting ACEP representation back to English...")
        english_text = convert_acep_to_english(acep_representation, context)
        print("\nEnglish representation:")
        print(english_text)
        
        # Create a simple reasoning trace for explanation
        reasoning_trace = {
            "session_id": "example_session",
            "timestamp": "2023-04-15T10:30:00Z",
            "steps": [
                {
                    "step_id": 1,
                    "pattern": "modus_ponens",
                    "premises": [
                        acep_representation["identifier"],
                        "example_stock_low_pe_ratio"
                    ],
                    "conclusion": "example_stock_undervalued",
                    "certainty": 0.8
                }
            ],
            "final_conclusions": [
                {
                    "identifier": "example_stock_undervalued",
                    "text": "The example stock is potentially undervalued",
                    "certainty": 0.8
                }
            ]
        }
        
        # Generate an explanation of the reasoning
        print("\nGenerating explanation from reasoning trace...")
        explanation_context = {
            "domain": "finance",
            "entity_id": "example_stock",
            "recommendation": "CONSIDER_BUY",
            "certainty": 0.8
        }
        explanation = generate_explanation(reasoning_trace, explanation_context)
        print("\nGenerated explanation:")
        print(explanation)
        
        # Demonstrate utility functions
        print("\nDemonstrating utility functions:")
        identifier = create_normalized_identifier(text)
        print(f"Normalized identifier: {identifier}")
        
        is_conditional = is_conditional_statement(text)
        print(f"Is conditional statement: {is_conditional}")
        
        # Create a vector embedding if possible
        try:
            print("\nCreating vector embedding...")
            embedding, metadata = create_embedding(text)
            print(f"Created embedding with {metadata['dimensions']} dimensions")
            print(f"Embedding shape: {embedding.shape}")
            print(f"First 5 values: {embedding[:5]}")
        except Exception as e:
            print(f"Could not create embedding: {str(e)}")
            print("Using deterministic vector as fallback...")
            vector = generate_deterministic_vector(text, 1536)
            print(f"Created deterministic vector with shape {vector.shape}")
            print(f"First 5 values: {vector[:5]}")
            
    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    example_usage()