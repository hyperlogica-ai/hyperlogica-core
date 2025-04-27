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
import backoff
from dotenv import load_dotenv
from openai import OpenAI

# Import vector operations
from .vector_operations import (
    generate_vector, normalize_vector, bind_vectors,
    unbind_vectors, bundle_vectors, calculate_similarity
)

# Load environment variables
load_dotenv() 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for API key
if "OPENAI_API_KEY" not in os.environ:
    logger.warning("OpenAI API key not found in environment variables.")

def get_english_to_acep_prompt(text: str, context: Dict[str, Any]) -> str:
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
    vector_dim = context.get("vector_dimension", 10000)
    
    prompt = f"""
    Convert this statement to ACEP (AI Conceptual Exchange Protocol) representation:

    Text: {text}
    Domain: {domain}
    Entity: {entity_id}
    Vector Dimension: {vector_dim}
    
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

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def convert_english_to_acep(text: str, context: Dict[str, Any], llm_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert English text to ACEP representation using an LLM and generate appropriate vector.
    
    Args:
        text (str): English text to convert
        context (dict): Contextual information about the domain and entity
        llm_options (dict): Options for the LLM API call
        
    Returns:
        dict: ACEP representation of the input text with vector representation
        
    Raises:
        ValueError: If the text cannot be converted to a valid ACEP representation
        Exception: If the API call fails after retries
    """
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Generate the prompt
    prompt = get_english_to_acep_prompt(text, context)
    
    # Log the conversion attempt
    logging.info(f"Converting to ACEP: {text[:50]}...")
    
    # Configure API call
    model = llm_options.get("model", "gpt-4")
    temperature = llm_options.get("temperature", 0.0)
    max_tokens = llm_options.get("max_tokens", 2000)
    
    logging.info(f"Calling OpenAI API with model: {model}")
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in AI knowledge representation and vector symbolic architectures."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        acep_representation = json.loads(content)
        
        # Validate required fields
        required_fields = ["type", "identifier", "attributes"]
        for field in required_fields:
            if field not in acep_representation:
                raise ValueError(f"Missing required field '{field}' in ACEP representation")
                
        # Ensure certainty is set
        if "certainty" not in acep_representation.get("attributes", {}):
            acep_representation["attributes"]["certainty"] = context.get("certainty", 0.9)
            
        # Ensure entity_id is set for concepts and facts
        if acep_representation["type"] == "concept" and "entity_id" not in acep_representation.get("attributes", {}):
            acep_representation["attributes"]["entity_id"] = context.get("entity_id", "")
            
        # Generate vector representation for the ACEP concept using identifier and content info
        vector_dimension = context.get("vector_dimension", 10000)
        vector_seed = hash(acep_representation["identifier"]) % (2**32)
        
        # Generate vector and add to representation
        vector = generate_vector(
            acep_representation["identifier"],
            dimension=vector_dimension,
            seed=vector_seed
        )
        
        acep_representation["vector"] = vector
        
        logging.info(f"Successfully converted to ACEP: {acep_representation['identifier']}")
        return acep_representation
        
    except Exception as e:
        logging.error(f"Error converting text to ACEP: {str(e)}")
        raise

def get_acep_to_english_prompt(acep_representation: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Generate a prompt for converting ACEP representation to natural language.
    
    Args:
        acep_representation (dict): ACEP structured representation
        context (dict): Contextual information for conversion
        
    Returns:
        str: A formatted prompt for the LLM
    """
    domain = context.get("domain", "general")
    entity_id = context.get("entity_id", "")
    
    # Need to convert vector to a string summary since we can't directly include in the prompt
    vector_info = ""
    if "vector" in acep_representation:
        vector = acep_representation["vector"]
        vector_info = f"Vector dimension: {vector.shape[0]}, norm: {np.linalg.norm(vector):.4f}"
        # Make a copy without the vector for the prompt
        acep_clean = acep_representation.copy()
        acep_clean.pop("vector", None)
    else:
        acep_clean = acep_representation
    
    prompt = f"""
    Convert this ACEP (AI Conceptual Exchange Protocol) representation into natural language:
    
    ACEP Representation:
    {json.dumps(acep_clean, indent=2)}
    
    Context:
    - Domain: {domain}
    - Entity ID: {entity_id}
    - {vector_info}
    
    Based on this ACEP representation, generate clear, concise natural language that:
    - Accurately conveys the same meaning
    - Includes appropriate qualifiers to express certainty
    - Uses domain-appropriate terminology
    - Is suitable for human readers
    
    Your response should be a single paragraph of natural language text that fully captures
    the information represented in the ACEP structure.
    """
    
    return prompt

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def convert_acep_to_english(acep_representation: Dict[str, Any], context: Dict[str, Any], llm_options: Dict[str, Any] = None) -> str:
    """
    Convert ACEP representation to English text using LLM.
    
    Args:
        acep_representation (dict): Structured ACEP representation to convert.
        context (dict): Contextual information about the domain and entity.
        llm_options (dict, optional): Options for the LLM API call.
                                     Defaults to None.
        
    Returns:
        str: Natural language English text representing the ACEP content.
        
    Raises:
        ValueError: If the ACEP representation is invalid or cannot be converted.
    """
    # Set default LLM options if not provided
    if llm_options is None:
        llm_options = {
            "model": "gpt-4",
            "temperature": 0.3,  # Slightly higher temperature for more natural language
            "max_tokens": 1000
        }
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Get the identifier for logging
    identifier = acep_representation.get("identifier", "unknown")
    logger.info(f"Converting ACEP to English: {identifier}")
    
    # Create prompt
    prompt = get_acep_to_english_prompt(acep_representation, context)
    
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
        raise

def get_explanation_prompt(reasoning_trace: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Generate a prompt for creating a natural language explanation from a reasoning trace.
    
    Args:
        reasoning_trace (dict): Reasoning trace data
        context (dict): Contextual information for the explanation
        
    Returns:
        str: A formatted prompt for the LLM
    """
    domain = context.get("domain", "general")
    entity_id = context.get("entity_id", "")
    recommendation = context.get("recommendation", "")
    certainty = context.get("certainty", 0.5)
    
    # Clean the trace of any vector data for the prompt
    def remove_vectors(obj):
        if isinstance(obj, dict):
            return {k: remove_vectors(v) for k, v in obj.items() if k != "vector"}
        elif isinstance(obj, list):
            return [remove_vectors(item) for item in obj]
        else:
            return obj
    
    clean_trace = remove_vectors(reasoning_trace)
    
    prompt = f"""
    Explain the following reasoning process in clear, natural language:
    
    Reasoning trace:
    {json.dumps(clean_trace, indent=2)}
    
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
    
    Format the explanation as a well-structured paragraph that clearly explains the reasoning process
    behind the recommendation, highlighting the most important evidence and how certainty was determined.
    """
    
    return prompt

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
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
            "model": "gpt-4",
            "temperature": 0.4,  # Higher temperature for more creative explanations
            "max_tokens": 1500
        }
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Create the prompt
    prompt = get_explanation_prompt(reasoning_trace, context)
    
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

def create_vector_embedding(text: str, model: str = "text-embedding-ada-002") -> Tuple[np.ndarray, Dict[str, Any]]:
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

def create_acep_relation(source_id: str, target_id: str, relation_type: str, 
                        attributes: Dict[str, Any], vector_dimension: int = 10000) -> Dict[str, Any]:
    """
    Create an ACEP relation representation between two concepts.
    
    Args:
        source_id (str): Source concept identifier
        target_id (str): Target concept identifier
        relation_type (str): Type of relation
        attributes (dict): Additional attributes for the relation
        vector_dimension (int, optional): Dimension for vector. Defaults to 10000.
        
    Returns:
        dict: ACEP relation representation
    """
    # Create a unique identifier for the relation
    relation_id = f"{source_id}_{relation_type}_{target_id}"
    
    # Ensure certainty is present
    if "certainty" not in attributes:
        attributes["certainty"] = 0.9  # Default high certainty
    
    # Generate a vector for the relation (in a real implementation, this would
    # be derived from source and target vectors, but we'll generate it here)
    relation_vector = generate_vector(relation_id, vector_dimension)
    
    # Create the relation representation
    relation = {
        "type": "relation",
        "identifier": relation_id,
        "source": source_id,
        "target": target_id,
        "relation_type": relation_type,
        "vector": relation_vector,
        "attributes": attributes
    }
    
    return relation

def create_acep_operation(operation_type: str, parameters: Dict[str, Any], 
                         attributes: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create an ACEP operation representation.
    
    Args:
        operation_type (str): Type of operation
        parameters (dict): Parameters for the operation
        attributes (dict, optional): Additional attributes. Defaults to None.
        
    Returns:
        dict: ACEP operation representation
    """
    if attributes is None:
        attributes = {}
    
    # Create a unique identifier for the operation
    operation_id = f"operation_{operation_type}_{int(time.time())}"
    
    # Ensure certainty is present
    if "certainty" not in attributes:
        attributes["certainty"] = 1.0  # Default high certainty for operations
    
    # Create the operation representation
    operation = {
        "type": "operation",
        "identifier": operation_id,
        "operation": operation_type,
        "parameters": parameters,
        "attributes": attributes
    }
    
    return operation

def create_semantic_pointer(text: str, vector_dimension: int = 10000) -> Dict[str, Any]:
    """
    Create a semantic pointer in ACEP format from text.
    
    Args:
        text (str): Text to represent
        vector_dimension (int, optional): Dimension for vector. Defaults to 10000.
        
    Returns:
        dict: ACEP semantic pointer representation
    """
    # Create a normalized identifier
    identifier = text.lower().replace(" ", "_")
    identifier = re.sub(r'[^\w]', '', identifier)
    
    if len(identifier) > 50:
        identifier = identifier[:50]
    
    # Generate a vector for the concept
    vector = generate_vector(identifier, vector_dimension)
    
    # Create the ACEP representation
    semantic_pointer = {
        "type": "concept",
        "identifier": identifier,
        "vector": vector,
        "attributes": {
            "text": text,
            "certainty": 1.0
        }
    }
    
    return semantic_pointer
