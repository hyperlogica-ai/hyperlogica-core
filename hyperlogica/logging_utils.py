"""
Hyperlogica Logging Utilities

This module provides a functional approach to logging for the Hyperlogica system.
It handles initialization of loggers, specialized logging for vector operations,
reasoning steps, and LLM interactions, and provides functionality to export
reasoning traces for analysis.

The logging system is designed to be configurable, allowing for different levels
of detail in logs and different output formats.
"""

import os
import json
import logging
from datetime import datetime
import time
from typing import Dict, List, Any, Optional, Callable, Union
import numpy as np

# Custom JSON encoder to handle NumPy arrays and other special types
class HyperlogicaJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.size > 10:  # Only show a few elements for large arrays
                return f"ndarray(shape={obj.shape}, sample=[{', '.join(map(str, obj.flatten()[:3]))}...])"
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def create_directory_if_not_exists(path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path (str): The directory path to create.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def initialize_logger(log_path: str, log_level: str) -> logging.Logger:
    """
    Initialize the logging system.
    
    Args:
        log_path (str): Path where log files should be stored.
        log_level (str): Minimum log level to record. Options include:
                         "debug", "info", "warning", "error".
        
    Returns:
        logging.Logger: Configured logger object.
    """
    # Create directory if it doesn't exist
    os.makedirs(log_path, exist_ok=True)
    
    # Configure level
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    log_level = level_map.get(log_level.lower(), logging.INFO)
    
    # Configure logger
    logger = logging.getLogger("hyperlogica.main")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    log_file = os.path.join(log_path, f"hyperlogica_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                         datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info("Logging system initialized with level: %s", log_level)
    
    return logger

def log_vector_operation(
    logger_config: Dict[str, Any],
    operation_type: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a vector operation.
    
    Args:
        logger_config (Dict[str, Any]): Logger configuration from initialize_logger.
        operation_type (str): Type of vector operation (e.g., "bind", "bundle", "permute").
        inputs (Dict[str, Any]): Dictionary of input vectors and parameters.
        outputs (Dict[str, Any]): Dictionary of output vectors and results.
        metadata (Dict[str, Any], optional): Additional metadata about the operation
                                           such as timing or caller info.
        
    Returns:
        None
    """
    if not logger_config["settings"]["include_vector_operations"]:
        return
    
    if "vector" not in logger_config:
        logger_config["main"].warning("Vector logging requested but not configured")
        return
    
    metadata = metadata or {}
    current_time = datetime.datetime.now().isoformat()
    
    # Prepare log entry
    log_entry = {
        "timestamp": current_time,
        "operation_type": operation_type,
        "inputs": _prepare_vectors_for_logging(inputs),
        "outputs": _prepare_vectors_for_logging(outputs),
        "metadata": metadata
    }
    
    try:
        log_message = json.dumps(log_entry, cls=HyperlogicaJSONEncoder)
        logger_config["vector"].debug(log_message)
    except Exception as e:
        logger_config["main"].error(f"Failed to log vector operation: {str(e)}")

def log_reasoning_step(
    logger_config: Dict[str, Any],
    pattern: str,
    premises: List[Any],
    conclusion: Dict[str, Any],
    certainty: float,
    entity_id: Optional[str] = None
) -> None:
    """
    Log a reasoning step.
    
    Args:
        logger_config (Dict[str, Any]): Logger configuration from initialize_logger.
        pattern (str): Reasoning pattern used (e.g., "modus_ponens", "conjunction").
        premises (List[Any]): List of premise representations used in the reasoning step.
        conclusion (Dict[str, Any]): Derived conclusion representation.
        certainty (float): Calculated certainty for the conclusion.
        entity_id (str, optional): Identifier for the entity being reasoned about.
        
    Returns:
        None
    """
    if not logger_config["settings"]["include_reasoning_steps"]:
        return
    
    if "reasoning" not in logger_config:
        logger_config["main"].warning("Reasoning logging requested but not configured")
        return
    
    current_time = datetime.datetime.now().isoformat()
    step_id = len(logger_config["reasoning_trace"]) + 1
    
    # Create a reasoning step entry
    step_entry = {
        "step_id": step_id,
        "timestamp": current_time,
        "entity_id": entity_id,
        "pattern": pattern,
        "premises": _clean_premises_for_logging(premises),
        "conclusion": _clean_conclusion_for_logging(conclusion),
        "certainty": certainty
    }
    
    # Store in the trace for later export
    logger_config["reasoning_trace"].append(step_entry)
    
    try:
        # Log a simplified version to the log file
        logger_config["reasoning"].info(
            f"Step {step_id}: {pattern} - Entity: {entity_id} - "
            f"Premises: {_summarize_premises(premises)} - "
            f"Conclusion: {_get_conclusion_identifier(conclusion)} - "
            f"Certainty: {certainty:.4f}"
        )
    except Exception as e:
        logger_config["main"].error(f"Failed to log reasoning step: {str(e)}")

def log_llm_interaction(
    logger_config: Dict[str, Any],
    prompt: str,
    response: Dict[str, Any],
    processing_time: float,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an interaction with an LLM.
    
    Args:
        logger_config (Dict[str, Any]): Logger configuration from initialize_logger.
        prompt (str): Prompt sent to the LLM API.
        response (Dict[str, Any]): Response received from the LLM API.
        processing_time (float): Time in seconds taken for the API call and processing.
        metadata (Dict[str, Any], optional): Additional metadata about the interaction.
        
    Returns:
        None
    """
    if not logger_config["settings"]["include_llm_interactions"]:
        return
    
    if "llm" not in logger_config:
        logger_config["main"].warning("LLM logging requested but not configured")
        return
    
    metadata = metadata or {}
    current_time = datetime.datetime.now().isoformat()
    
    # Prepare log entry - truncate long prompts/responses for readability
    MAX_PROMPT_LENGTH = 500
    truncated_prompt = prompt[:MAX_PROMPT_LENGTH] + "..." if len(prompt) > MAX_PROMPT_LENGTH else prompt
    
    # Extract just the text content from response for brevity
    response_content = _extract_response_content(response)
    MAX_RESPONSE_LENGTH = 500
    truncated_response = response_content[:MAX_RESPONSE_LENGTH] + "..." if len(response_content) > MAX_RESPONSE_LENGTH else response_content
    
    try:
        # Log summary to the log file
        logger_config["llm"].info(
            f"LLM interaction - Time: {processing_time:.2f}s - "
            f"Prompt: {truncated_prompt} - "
            f"Response: {truncated_response} - "
            f"Metadata: {json.dumps(metadata, cls=HyperlogicaJSONEncoder)}"
        )
        
        # For detailed logging, save the full interaction to a separate file
        if logger_config["settings"]["log_level"].lower() == "debug":
            interaction_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            interaction_file = os.path.join(
                logger_config["log_path"], 
                f"llm_interaction_{interaction_id}.json"
            )
            
            with open(interaction_file, 'w') as f:
                json.dump({
                    "timestamp": current_time,
                    "processing_time": processing_time,
                    "prompt": prompt,
                    "response": response,
                    "metadata": metadata
                }, f, cls=HyperlogicaJSONEncoder, indent=2)
            
            logger_config["llm"].debug(f"Full interaction saved to {interaction_file}")
    except Exception as e:
        logger_config["main"].error(f"Failed to log LLM interaction: {str(e)}")

def export_reasoning_trace(
    logger_config: Dict[str, Any],
    format: str = "json",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export the reasoning trace in the specified format.
    
    Args:
        logger_config (Dict[str, Any]): Logger configuration from initialize_logger.
        format (str, optional): Output format such as "json", "text", or "graph". 
                              Defaults to "json".
        output_path (str, optional): Path where the trace should be saved.
                                   If None, the trace is not saved to disk.
        
    Returns:
        Dict[str, Any]: Structured representation of the reasoning trace.
        
    Raises:
        ValueError: If an unsupported export format is specified.
    """
    if format.lower() not in ["json", "text", "graph"]:
        raise ValueError(f"Unsupported export format: {format}. Supported formats: json, text, graph")
    
    if not logger_config["reasoning_trace"]:
        logger_config["main"].warning("No reasoning steps to export")
        return {"steps": [], "session_id": logger_config["session_id"]}
    
    trace = {
        "session_id": logger_config["session_id"],
        "timestamp": datetime.datetime.now().isoformat(),
        "steps": logger_config["reasoning_trace"],
        "format": format
    }
    
    # Save to disk if output_path is provided
    if output_path:
        try:
            create_directory_if_not_exists(output_path)
            
            if format.lower() == "json":
                output_file = os.path.join(output_path, f"reasoning_trace_{logger_config['session_id']}.json")
                with open(output_file, 'w') as f:
                    json.dump(trace, f, cls=HyperlogicaJSONEncoder, indent=2)
            elif format.lower() == "text":
                output_file = os.path.join(output_path, f"reasoning_trace_{logger_config['session_id']}.txt")
                with open(output_file, 'w') as f:
                    f.write(_format_trace_as_text(trace))
            elif format.lower() == "graph":
                output_file = os.path.join(output_path, f"reasoning_trace_{logger_config['session_id']}.dot")
                with open(output_file, 'w') as f:
                    f.write(_format_trace_as_graph(trace))
            
            logger_config["main"].info(f"Reasoning trace exported to {output_file}")
        except Exception as e:
            logger_config["main"].error(f"Failed to export reasoning trace: {str(e)}")
    
    return trace

def log_performance_metrics(
    logger_config: Dict[str, Any],
    operation: str,
    execution_time: float,
    metrics: Dict[str, Any]
) -> None:
    """
    Log performance metrics for an operation.
    
    Args:
        logger_config (Dict[str, Any]): Logger configuration from initialize_logger.
        operation (str): Name of the operation being measured.
        execution_time (float): Execution time in seconds.
        metrics (Dict[str, Any]): Additional metrics specific to the operation.
        
    Returns:
        None
    """
    logger_config["main"].info(
        f"Performance - {operation} - Time: {execution_time:.4f}s - "
        f"Metrics: {json.dumps(metrics, cls=HyperlogicaJSONEncoder)}"
    )

def timer(logger_config: Dict[str, Any], operation_name: str) -> Callable:
    """
    Function decorator to time and log the execution of functions.
    
    Args:
        logger_config (Dict[str, Any]): Logger configuration from initialize_logger.
        operation_name (str): Name of the operation to log.
        
    Returns:
        Callable: Decorator function that times and logs the execution.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            log_performance_metrics(
                logger_config,
                operation_name,
                execution_time,
                {"function": func.__name__}
            )
            
            return result
        return wrapper
    return decorator

# Helper functions for formatting log entries

def _prepare_vectors_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare vector data for logging, trimming large arrays for readability.
    
    Args:
        data (Dict[str, Any]): Dictionary containing vector data.
        
    Returns:
        Dict[str, Any]: Processed data suitable for logging.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.size > 10:  # Summarize large arrays
                result[key] = {
                    "shape": value.shape,
                    "type": str(value.dtype),
                    "sample": value.flatten()[:3].tolist(),
                    "norm": float(np.linalg.norm(value)) if value.size > 0 else 0
                }
            else:
                result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = _prepare_vectors_for_logging(value)
        else:
            result[key] = value
    return result

def _clean_premises_for_logging(premises: List[Any]) -> List[Dict[str, Any]]:
    """
    Clean premise data for logging, extracting just the essential information.
    
    Args:
        premises (List[Any]): List of premise representations.
        
    Returns:
        List[Dict[str, Any]]: Cleaned premises suitable for logging.
    """
    cleaned_premises = []
    for premise in premises:
        if isinstance(premise, dict):
            # Extract essential information for logging
            cleaned_premise = {
                "identifier": premise.get("identifier", "unknown"),
                "type": premise.get("type", "unknown")
            }
            
            if "certainty" in premise:
                cleaned_premise["certainty"] = premise["certainty"]
                
            if "content" in premise:
                cleaned_premise["content"] = {
                    k: v for k, v in premise["content"].items() 
                    if k not in ["vector"]
                }
                
            cleaned_premises.append(cleaned_premise)
        else:
            # If it's not a dictionary, just use it as is
            cleaned_premises.append(premise)
    
    return cleaned_premises

def _clean_conclusion_for_logging(conclusion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean conclusion data for logging, extracting just the essential information.
    
    Args:
        conclusion (Dict[str, Any]): Conclusion representation.
        
    Returns:
        Dict[str, Any]: Cleaned conclusion suitable for logging.
    """
    if not isinstance(conclusion, dict):
        return {"value": conclusion}
        
    cleaned_conclusion = {
        "identifier": conclusion.get("identifier", "unknown"),
        "type": conclusion.get("type", "unknown")
    }
    
    if "certainty" in conclusion:
        cleaned_conclusion["certainty"] = conclusion["certainty"]
        
    if "content" in conclusion:
        cleaned_conclusion["content"] = {
            k: v for k, v in conclusion["content"].items()
            if k not in ["vector"]
        }
    
    return cleaned_conclusion

def _summarize_premises(premises: List[Any]) -> str:
    """
    Create a summary string for premises for compact logging.
    
    Args:
        premises (List[Any]): List of premise representations.
        
    Returns:
        str: Short summary of premises for logging.
    """
    premise_ids = []
    for premise in premises:
        if isinstance(premise, dict):
            premise_id = premise.get("identifier", "unknown")
            premise_ids.append(premise_id)
        else:
            premise_ids.append(str(premise))
    
    return "[" + ", ".join(premise_ids) + "]"

def _get_conclusion_identifier(conclusion: Dict[str, Any]) -> str:
    """
    Extract conclusion identifier for compact logging.
    
    Args:
        conclusion (Dict[str, Any]): Conclusion representation.
        
    Returns:
        str: Conclusion identifier for logging.
    """
    if not isinstance(conclusion, dict):
        return str(conclusion)
    
    return conclusion.get("identifier", "unknown")

def _extract_response_content(response: Dict[str, Any]) -> str:
    """
    Extract text content from an LLM API response.
    
    Args:
        response (Dict[str, Any]): Response received from the LLM API.
        
    Returns:
        str: Extracted text content.
    """
    # Handle different LLM API response formats
    if isinstance(response, str):
        return response
    
    if isinstance(response, dict):
        # OpenAI format
        if "choices" in response and response["choices"]:
            if isinstance(response["choices"][0], dict):
                if "text" in response["choices"][0]:
                    return response["choices"][0]["text"]
                elif "message" in response["choices"][0] and "content" in response["choices"][0]["message"]:
                    return response["choices"][0]["message"]["content"]
        
        # Simple content field
        if "content" in response:
            return response["content"]
        
        # Other possible formats
        for field in ["text", "output", "result", "generated_text"]:
            if field in response:
                return response[field]
    
    # Fallback: serialize the whole response
    try:
        return json.dumps(response, cls=HyperlogicaJSONEncoder)
    except:
        return str(response)

def _format_trace_as_text(trace: Dict[str, Any]) -> str:
    """
    Format the reasoning trace as a text document.
    
    Args:
        trace (Dict[str, Any]): Reasoning trace to format.
        
    Returns:
        str: Formatted text representation.
    """
    lines = [
        f"Reasoning Trace: Session {trace['session_id']}",
        f"Generated: {trace['timestamp']}",
        "=" * 80,
        ""
    ]
    
    for step in trace["steps"]:
        lines.append(f"Step {step['step_id']}: {step['pattern']}")
        if step.get("entity_id"):
            lines.append(f"Entity: {step['entity_id']}")
        
        lines.append("Premises:")
        for i, premise in enumerate(step["premises"]):
            if isinstance(premise, dict) and "identifier" in premise:
                lines.append(f"  {i+1}. {premise['identifier']} (certainty: {premise.get('certainty', 'unknown')})")
            else:
                lines.append(f"  {i+1}. {premise}")
        
        lines.append("Conclusion:")
        conclusion = step["conclusion"]
        if isinstance(conclusion, dict) and "identifier" in conclusion:
            lines.append(f"  {conclusion['identifier']} (certainty: {step['certainty']:.4f})")
        else:
            lines.append(f"  {conclusion} (certainty: {step['certainty']:.4f})")
        
        lines.append("")
        lines.append("-" * 40)
        lines.append("")
    
    return "\n".join(lines)

def _format_trace_as_graph(trace: Dict[str, Any]) -> str:
    """
    Format the reasoning trace as a Graphviz DOT graph.
    
    Args:
        trace (Dict[str, Any]): Reasoning trace to format.
        
    Returns:
        str: DOT graph representation.
    """
    lines = [
        'digraph ReasoningTrace {',
        '  rankdir=TB;',
        '  node [shape=box, style=filled, fillcolor=lightblue];',
        '  edge [fontsize=10];',
        ''
    ]
    
    # Create nodes for premises and conclusions
    node_ids = set()
    
    for step in trace["steps"]:
        step_id = step["step_id"]
        
        # Add premise nodes
        for i, premise in enumerate(step["premises"]):
            premise_id = None
            if isinstance(premise, dict) and "identifier" in premise:
                premise_id = premise["identifier"]
                certainty = premise.get("certainty", "unknown")
            else:
                premise_id = str(premise)
                certainty = "unknown"
            
            safe_id = premise_id.replace("-", "_").replace(".", "_")
            if safe_id not in node_ids:
                node_ids.add(safe_id)
                lines.append(f'  {safe_id} [label="{premise_id}\\nCertainty: {certainty}"];')
        
        # Add conclusion node
        conclusion = step["conclusion"]
        if isinstance(conclusion, dict) and "identifier" in conclusion:
            conclusion_id = conclusion["identifier"]
        else:
            conclusion_id = str(conclusion)
        
        safe_conclusion_id = conclusion_id.replace("-", "_").replace(".", "_")
        certainty = step["certainty"]
        
        if safe_conclusion_id not in node_ids:
            node_ids.add(safe_conclusion_id)
            lines.append(f'  {safe_conclusion_id} [label="{conclusion_id}\\nCertainty: {certainty:.4f}", fillcolor=lightgreen];')
        
        # Add invisible step node to group premises
        step_node = f"step_{step_id}"
        lines.append(f'  {step_node} [label="{step["pattern"]}", shape=ellipse, fillcolor=lightyellow];')
        
        # Connect premises to step
        for premise in step["premises"]:
            if isinstance(premise, dict) and "identifier" in premise:
                premise_id = premise["identifier"]
            else:
                premise_id = str(premise)
            
            safe_id = premise_id.replace("-", "_").replace(".", "_")
            lines.append(f'  {safe_id} -> {step_node};')
        
        # Connect step to conclusion
        lines.append(f'  {step_node} -> {safe_conclusion_id} [label="certainty: {certainty:.4f}"];')
    
    lines.append('}')
    return "\n".join(lines)
