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

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def convert_english_to_acep(text: str, context: Dict[str, Any], llm_options: Dict[str, Any]) -> Dict[str, Any]:
    """Convert English text to ACEP representation using an LLM and generate vector."""
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Set up context values
    domain = context.get("domain", "general")
    certainty = context.get("certainty", 0.9)
    entity_id = context.get("entity_id", "")
    vector_dim = context.get("vector_dimension", 10000)
    
    # Create prompt for ACEP angle-bracket syntax
    prompt = f"""
    Convert this text to ACEP (AI Conceptual Exchange Protocol) representation using angle-bracket syntax:
    
    Text: {text}
    Domain: {domain}
    Entity: {entity_id}
    
    For conditional statements (if-then), use the format:
    <{{concept:condition}}> → <{{causal:{certainty}}}> → <{{concept:result}}>
    
    For facts or statements, use:
    <{{concept:fact, certainty:{certainty}}}>
    
    Example for "If it rains, the ground gets wet":
    <{{concept:rain}}> → <{{causal:0.9}}> → <{{concept:ground_wet}}>
    
    Include all relevant attributes and ensure proper ACEP syntax with angle brackets.
    """
    
    # Configure API call
    model = llm_options.get("model", "gpt-4")
    temperature = llm_options.get("temperature", 0.0)
    max_tokens = llm_options.get("max_tokens", 2000)
    
    logging.info(f"Converting to ACEP: {text[:50]}...")
    
    try:
        # Make the API call without JSON format constraint
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
        
        # Generate an identifier based on the text
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        identifier = f"{entity_id}_{text_hash}" if entity_id else f"concept_{text_hash}"
        
        # Generate vector for this concept
        vector_dimension = context.get("vector_dimension", 10000)
        vector = generate_vector(identifier, dimension=vector_dimension)
        
        # Create a structured representation that includes the ACEP text
        acep_representation = {
            "type": "acep_concept",
            "identifier": identifier,
            "acep_text": acep_text,
            "original_text": text,
            "attributes": {
                "certainty": certainty,
                "entity_id": entity_id,
                "domain": domain
            },
            "vector": vector
        }
        
        # Determine if this is a conditional (rule) or fact
        if "→" in acep_text or "->" in acep_text:
            acep_representation["type"] = "acep_relation"
            # Extract conditional parts if possible
            try:
                if "if" in text.lower() and "then" in text.lower():
                    parts = text.lower().split("then")
                    antecedent = parts[0].replace("if", "", 1).strip()
                    consequent = parts[1].strip()
                    acep_representation["attributes"]["antecedent"] = antecedent
                    acep_representation["attributes"]["consequent"] = consequent
                    acep_representation["attributes"]["conditional"] = True
            except Exception as e:
                logging.warning(f"Could not extract conditional parts: {e}")
        
        logging.info(f"Successfully converted to ACEP: {identifier}")
        return acep_representation
        
    except Exception as e:
        logging.error(f"Error converting text to ACEP: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
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
