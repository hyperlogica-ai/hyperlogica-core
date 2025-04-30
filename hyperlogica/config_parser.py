"""
Configuration Parser for Hyperlogica System.

This module handles parsing, validation, and extraction of configuration elements
from the input JSON file for the Hyperlogica reasoning system, focused on proper ACEP
representation. This simplified version has removed English-to-ACEP conversion components.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Union
from jsonschema import validate, ValidationError
from .error_handling import success, error, is_success, is_error, get_value, get_error

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
                "include_reasoning_steps": {"type": "boolean"}
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
                        "required": ["acep"],
                        "properties": {
                            "acep": {"type": "object"},
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
                                    "required": ["acep"],
                                    "properties": {
                                        "acep": {"type": "object"},
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
        
        logger.info(f"Successfully parsed configuration file: {input_path}")
        return success(config)  # This returns (config, None)
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
        return error(f"Invalid configuration: {e}")
    
    # Add default values if not present
    config_dict.setdefault("persistence", {})
    config_dict["persistence"].setdefault("load_previous_state", False)
    config_dict["persistence"].setdefault("save_state", True)
    
    config_dict.setdefault("logging", {})
    config_dict["logging"].setdefault("log_level", "info")
    config_dict["logging"].setdefault("include_vector_operations", False)
    config_dict["logging"].setdefault("include_reasoning_steps", True)
    
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
    return success(config_dict)

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

def extract_rules(config_dict: Dict[str, Any]) -> list:
    """
    Extract rule ACEP representations from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        list: List of rule ACEP dictionaries.
        
    Raises:
        ValueError: If no rules are found.
    """
    if "input_data" not in config_dict or "rules" not in config_dict["input_data"]:
        logger.error("No rules found in configuration")
        raise ValueError("No rules found in configuration")
    
    rules = []
    for rule_data in config_dict["input_data"]["rules"]:
        # Extract the ACEP representation and certainty
        acep_repr = rule_data.get("acep", {})
        certainty = rule_data.get("certainty", 0.9)
        
        # Ensure it has certainty in attributes
        if "attributes" in acep_repr:
            acep_repr["attributes"]["certainty"] = certainty
        else:
            acep_repr["attributes"] = {"certainty": certainty}
        
        rules.append(acep_repr)
    
    logger.info(f"Extracted {len(rules)} rule ACEP representations from configuration")
    return rules

def extract_entities(config_dict: Dict[str, Any]) -> list:
    """
    Extract entities with fact ACEP representations from configuration.
    
    Args:
        config_dict (dict): Validated configuration dictionary.
        
    Returns:
        list: List of entity dictionaries with ACEP facts.
    """
    if "input_data" not in config_dict or "entities" not in config_dict["input_data"]:
        logger.warning("No entities found in configuration")
        return []
    
    entities = []
    for entity_data in config_dict["input_data"]["entities"]:
        entity_id = entity_data.get("id", "")
        entity_name = entity_data.get("name", entity_id)
        
        # Process facts for this entity
        facts = []
        for fact_data in entity_data.get("facts", []):
            # Extract the ACEP representation and certainty
            acep_repr = fact_data.get("acep", {})
            certainty = fact_data.get("certainty", 0.8)
            
            # Ensure it has certainty and entity_id in attributes
            if "attributes" in acep_repr:
                acep_repr["attributes"]["certainty"] = certainty
                acep_repr["attributes"]["entity_id"] = entity_id
            else:
                acep_repr["attributes"] = {
                    "certainty": certainty,
                    "entity_id": entity_id
                }
            
            facts.append(acep_repr)
        
        # Create entity with processed facts
        entity = {
            "id": entity_id,
            "name": entity_name,
            "facts": facts
        }
        
        entities.append(entity)
    
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
    raw_config_result = parse_input_config(input_path)
    if is_error(raw_config_result):
        raise ValueError(get_error(raw_config_result))
    
    raw_config = get_value(raw_config_result)
    
    # Validate and add defaults
    validated_config_result = validate_config(raw_config)
    if is_error(validated_config_result):
        raise ValueError(get_error(validated_config_result))
    
    validated_config = get_value(validated_config_result)
    
    # Extract components
    processed_config = {
        "processing_options": extract_processing_options(validated_config),
        "persistence_options": extract_persistence_options(validated_config),
        "output_schema": extract_output_schema(validated_config),
        "rules": extract_rules(validated_config),
        "entities": extract_entities(validated_config),
        "raw_config": validated_config  # Include the full validated config for reference
    }
    
    logger.info("Configuration processing complete")
    return processed_config
