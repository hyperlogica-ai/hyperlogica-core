"""
Configuration Parser for Hyperlogica System.

This module handles parsing, validation, and extraction of configuration elements
from the input JSON file for the Hyperlogica reasoning system.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, Union
from jsonschema import validate, ValidationError

# Set up logging
logger = logging.getLogger(__name__)

# Configuration schema for validation
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["processing", "input_data"],
    "properties": {
        "processing": {
            "type": "object",
            "required": ["vector_dimension", "reasoning_approach"],
            "properties": {
                "vector_dimension": {"type": "integer", "minimum": 100},
                "vector_type": {"type": "string", "enum": ["binary", "continuous"]},
                "reasoning_approach": {"type": "string"},
                "certainty_propagation": {"type": "string", "enum": ["min", "product", "noisy_or", "weighted"]},
                "recalibration_enabled": {"type": "boolean"},
                "max_reasoning_depth": {"type": "integer", "minimum": 1},
                "domain_config": {"type": "object"}
            }
        },
        "persistence": {
            "type": "object",
            "properties": {
                "load_previous_state": {"type": "boolean"},
                "previous_state_path": {"type": "string"},
                "save_state": {"type": "boolean"},
                "state_save_path": {"type": "string"}
            }
        },
        "logging": {
            "type": "object",
            "properties": {
                "log_level": {"type": "string", "enum": ["debug", "info", "warning", "error"]},
                "log_path": {"type": "string"},
                "include_vector_operations": {"type": "boolean"},
                "include_llm_interactions": {"type": "boolean"},
                "include_reasoning_steps": {"type": "boolean"}
            }
        },
        "llm": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 1},
                "max_tokens": {"type": "integer", "minimum": 1}
            }
        },
        "input_data": {
            "type": "object",
            "required": ["rules"],
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["text"],
                        "properties": {
                            "text": {"type": "string"},
                            "certainty": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "facts"],
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "facts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["text"],
                                    "properties": {
                                        "text": {"type": "string"},
                                        "certainty": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"}
                        }
                    }
                },
                "include_reasoning_trace": {"type": "boolean"},
                "include_vector_details": {"type": "boolean"}
            }
        }
    }
}

def parse_input_config(input_path: str) -> Dict[str, Any]:
    """
    Parse input JSON configuration file.
    
    Args:
        input_path (str): Path to the JSON configuration file.
        
    Returns:
        dict: Parsed configuration dictionary containing all settings and data.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        if not os.path.exists(input_path):
            return error(f"Configuration file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"Successfully parsed configuration file: {input_path}")
        return success(config)  # This should return (config, None)
    except json.JSONDecodeError as e:
        return error(f"Invalid JSON in configuration file: {str(e)}")
    except Exception as e:
        return error(f"Error parsing configuration: {str(e)}")

def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary and provide defaults for missing values.
    
    Args:
        config_dict (dict): Raw configuration dictionary from parsed JSON.
        
    Returns:
        dict: Validated configuration with defaults applied where necessary.
        
    Raises:
        ValueError: If required configuration fields are missing or invalid.
    """
    # Validate against schema
    try:
        validate(instance=config_dict, schema=CONFIG_SCHEMA)
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise ValueError(f"Invalid configuration: {e}")
    
    # Add default values if not present
    config_dict.setdefault("persistence", {})
    config_dict["persistence"].setdefault("load_previous_state", False)
    config_dict["persistence"].setdefault("save_state", True)
    
    config_dict.setdefault("logging", {})
    config_dict["logging"].setdefault("log_level", "info")
    config_dict["logging"].setdefault("include_vector_operations", False)
    config_dict["logging"].setdefault("include_llm_interactions", True)
    config_dict["logging"].setdefault("include_reasoning_steps", True)
    
    config_dict.setdefault("llm", {})
    config_dict["llm"].setdefault("model", "gpt-3.5-turbo")
    config_dict["llm"].setdefault("temperature", 0.0)
    config_dict["llm"].setdefault("max_tokens", 2000)
    
    config_dict.setdefault("output_schema", {})
    config_dict["output_schema"].setdefault("format", "json")
    config_dict["output_schema"].setdefault("include_reasoning_trace", True)
    config_dict["output_schema"].setdefault("include_vector_details", False)
    
    # Add defaults to processing
    config_dict["processing"].setdefault("vector_type", "binary")
    config_dict["processing"].setdefault("certainty_propagation", "min")
    config_dict["processing"].setdefault("recalibration_enabled", True)
    config_dict["processing"].setdefault("max_reasoning_depth", 10)
    config_dict["processing"].setdefault("domain_config", {})
    
    # Ensure all rules and facts have certainty
    for rule in config_dict["input_data"]["rules"]:
        rule.setdefault("certainty", 0.9)  # Default high certainty for rules
    
    if "entities" in config_dict["input_data"]:
        for entity in config_dict["input_data"]["entities"]:
            for fact in entity["facts"]:
                fact.setdefault("certainty", 0.8)  # Default certainty for facts
    
    logger.info("Configuration validated and defaults applied")
    return config_dict

def extract_processing_options(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract processing-related options from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing only processing-related options such as 
             vector dimension, reasoning approach, and certainty propagation method.
    """
    if "processing" not in config_dict:
        logger.warning("No processing options found in configuration")
        return {}
    
    processing_options = config_dict["processing"].copy()
    logger.debug(f"Extracted processing options: {processing_options}")
    return processing_options

def extract_persistence_options(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract persistence-related options from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing persistence options such as save paths,
             load flags, and state management settings.
    """
    if "persistence" not in config_dict:
        logger.warning("No persistence options found in configuration")
        return {}
    
    persistence_options = config_dict["persistence"].copy()
    logger.debug(f"Extracted persistence options: {persistence_options}")
    return persistence_options

def extract_output_schema(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract output schema definition from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing output schema definition including field
             specifications, format settings, and inclusion flags.
    """
    if "output_schema" not in config_dict:
        logger.warning("No output schema found in configuration")
        return {}
    
    schema = config_dict["output_schema"].copy()
    logger.debug(f"Extracted output schema: {schema}")
    return schema

def extract_llm_options(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract LLM-related options from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing LLM options such as model name,
             temperature, and token limits.
    """
    if "llm" not in config_dict:
        logger.warning("No LLM options found in configuration")
        return {}
    
    llm_options = config_dict["llm"].copy()
    logger.debug(f"Extracted LLM options: {llm_options}")
    return llm_options

def extract_logging_options(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract logging-related options from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        dict: Dictionary containing logging options such as log level,
             log path, and inclusion flags.
    """
    if "logging" not in config_dict:
        logger.warning("No logging options found in configuration")
        return {}
    
    logging_options = config_dict["logging"].copy()
    logger.debug(f"Extracted logging options: {logging_options}")
    return logging_options

def extract_rules(config_dict: Dict[str, Any]) -> list:
    """
    Extract rules from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        list: List of rule dictionaries.
        
    Raises:
        ValueError: If no rules are found.
    """
    if "input_data" not in config_dict or "rules" not in config_dict["input_data"]:
        logger.error("No rules found in configuration")
        raise ValueError("No rules found in configuration")
    
    rules = config_dict["input_data"]["rules"]
    logger.info(f"Extracted {len(rules)} rules from configuration")
    return rules

def extract_entities(config_dict: Dict[str, Any]) -> list:
    """
    Extract entities from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        list: List of entity dictionaries.
    """
    if "input_data" not in config_dict or "entities" not in config_dict["input_data"]:
        logger.warning("No entities found in configuration")
        return []
    
    entities = config_dict["input_data"]["entities"]
    logger.info(f"Extracted {len(entities)} entities from configuration")
    return entities

def process_config_file(input_path: str) -> Dict[str, Any]:
    """
    Process configuration file from parsing to validation and extraction.
    
    Args:
        input_path (str): Path to the configuration file.
        
    Returns:
        dict: Dictionary with all processed configuration components.
        
    Raises:
        Various exceptions for file not found, invalid JSON, validation errors.
    """
    # Parse the configuration file
    raw_config = parse_input_config(input_path)
    
    # Validate and add defaults
    validated_config = validate_config(raw_config)
    
    # Extract components
    processed_config = {
        "processing_options": extract_processing_options(validated_config),
        "persistence_options": extract_persistence_options(validated_config),
        "output_schema": extract_output_schema(validated_config),
        "llm_options": extract_llm_options(validated_config),
        "logging_options": extract_logging_options(validated_config),
        "rules": extract_rules(validated_config),
        "entities": extract_entities(validated_config),
        "raw_config": validated_config  # Include the full validated config for reference
    }
    
    logger.info("Configuration processing complete")
    return processed_config
