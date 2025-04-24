# Hyperlogica API Reference

## Core Modules

### Configuration Parser (`config_parser.py`)

#### `parse_input_config(input_path: str) -> Result[Dict[str, Any], str]`

Parses a JSON configuration file from the given path.

- **Parameters**:
  - `input_path`: Path to the configuration file
- **Returns**:
  - `Result` containing the parsed configuration or an error

#### `validate_config(config: Dict[str, Any]) -> Dict[str, Any]`

Validates a configuration dictionary and applies default values.

- **Parameters**:
  - `config`: Configuration dictionary to validate
- **Returns**:
  - Validated configuration with defaults applied

#### `extract_processing_options(config: Dict[str, Any]) -> Dict[str, Any]`

Extracts processing-related options from the configuration.

- **Parameters**:
  - `config`: Validated configuration dictionary
- **Returns**:
  - Dictionary containing processing options

### Vector Operations (`vector_operations.py`)

#### `generate_vector(text: str, dimension: int, seed: Optional[int] = None) -> np.ndarray`

Generates a high-dimensional vector from text.

- **Parameters**:
  - `text`: Input text
  - `dimension`: Vector dimension
  - `seed`: Optional random seed for reproducibility
- **Returns**:
  - Normalized vector of the specified dimension

#### `normalize_vector(vector: np.ndarray) -> np.ndarray`

Normalizes a vector to unit length.

- **Parameters**:
  - `vector`: Input vector
- **Returns**:
  - Normalized vector with the same direction but unit length
- **Raises**:
  - `ValueError`: If the input is a zero vector

#### `bind_vectors(vector_a: np.ndarray, vector_b: np.ndarray, method: str = "xor") -> np.ndarray`

Binds two vectors to create a new associated vector.

- **Parameters**:
  - `vector_a`: First input vector
  - `vector_b`: Second input vector
  - `method`: Binding method: "xor" or "convolution"
- **Returns**:
  - Bound vector

#### `unbind_vectors(bound_vector: np.ndarray, vector_a: np.ndarray, method: str = "xor") -> np.ndarray`

Unbinds to recover vector_b from bound_vector and vector_a.

- **Parameters**:
  - `bound_vector`: The bound vector
  - `vector_a`: One of the original vectors
  - `method`: Unbinding method: "xor" or "convolution"
- **Returns**:
  - Recovered approximation of vector_b

#### `bundle_vectors(vector_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray`

Bundles multiple vectors with optional weighting.

- **Parameters**:
  - `vector_list`: List of vectors to bundle
  - `weights`: Optional weights for each vector
- **Returns**:
  - Bundled vector

### Vector Store (`vector_store.py`)

#### `create_store(dimension: int, index_type: str = "flat") -> Dict[str, Any]`

Creates a new vector store with FAISS.

- **Parameters**:
  - `dimension`: Dimensionality of vectors
  - `index_type`: Type of index: "flat", "ivf", or "hnsw"
- **Returns**:
  - Vector store dictionary

#### `add_vector(store: Dict[str, Any], identifier: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool`

Adds a vector to the store with metadata.

- **Parameters**:
  - `store`: Vector store
  - `identifier`: Unique identifier
  - `vector`: Vector to add
  - `metadata`: Additional metadata
- **Returns**:
  - `True` if successful

#### `get_vector(store: Dict[str, Any], identifier: str) -> Dict[str, Any]`

Retrieves a vector and its metadata by identifier.

- **Parameters**:
  - `store`: Vector store
  - `identifier`: Unique identifier
- **Returns**:
  - Dictionary containing the vector and metadata

#### `find_similar_vectors(store: Dict[str, Any], query_vector: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]`

Finds the most similar vectors to the query vector.

- **Parameters**:
  - `store`: Vector store
  - `query_vector`: Query vector
  - `top_n`: Number of results to return
- **Returns**:
  - List of similar vectors with metadata

### LLM Interface (`llm_interface.py`)

#### `convert_english_to_acep(text: str, context: Dict[str, Any], llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`

Converts English text to ACEP representation.

- **Parameters**:
  - `text`: English text
  - `context`: Context information
  - `llm_options`: Options for the LLM API
- **Returns**:
  - ACEP representation

#### `convert_acep_to_english(acep: Dict[str, Any], context: Dict[str, Any], llm_options: Optional[Dict[str, Any]] = None) -> str`

Converts ACEP representation to English text.

- **Parameters**:
  - `acep`: ACEP representation
  - `context`: Context information
  - `llm_options`: Options for the LLM API
- **Returns**:
  - English text

#### `generate_explanation(reasoning_trace: Dict[str, Any], context: Dict[str, Any], llm_options: Optional[Dict[str, Any]] = None) -> str`

Generates an explanation from a reasoning trace.

- **Parameters**:
  - `reasoning_trace`: Trace of reasoning steps
  - `context`: Context information
  - `llm_options`: Options for the LLM API
- **Returns**:
  - Natural language explanation

### Reasoning Engine (`reasoning_engine.py`)

#### `apply_modus_ponens(rule: Dict[str, Any], fact: Dict[str, Any], store: Dict[str, Any]) -> Dict[str, Any]`

Applies modus ponens: If P→Q and P, then Q.

- **Parameters**:
  - `rule`: Conditional rule representation
  - `fact`: Fact representation
  - `store`: Vector store
- **Returns**:
  - Derived conclusion

#### `apply_modus_tollens(rule: Dict[str, Any], negated_fact: Dict[str, Any], store: Dict[str, Any]) -> Dict[str, Any]`

Applies modus tollens: If P→Q and ¬Q, then ¬P.

- **Parameters**:
  - `rule`: Conditional rule representation
  - `negated_fact`: Negated fact representation
  - `store`: Vector store
- **Returns**:
  - Derived conclusion

#### `calculate_certainty(evidence_certainties: List[float], method: str) -> float`

Calculates overall certainty from multiple pieces of evidence.

- **Parameters**:
  - `evidence_certainties`: List of certainty values
  - `method`: Calculation method
- **Returns**:
  - Combined certainty value

### State Management (`state_management.py`)

#### `create_state(session_id: str) -> Dict[str, Any]`

Creates a new state for a session.

- **Parameters**:
  - `session_id`: Unique session identifier
- **Returns**:
  - New state dictionary

#### `add_concept_to_state(state: Dict[str, Any], concept: Dict[str, Any]) -> Dict[str, Any]`

Adds a concept to the state.

- **Parameters**:
  - `state`: Current state
  - `concept`: Concept to add
- **Returns**:
  - Updated state

#### `add_relation_to_state(state: Dict[str, Any], relation: Dict[str, Any]) -> Dict[str, Any]`

Adds a relation to the state.

- **Parameters**:
  - `state`: Current state
  - `relation`: Relation to add
- **Returns**:
  - Updated state

#### `resolve_reference(state: Dict[str, Any], reference: str) -> Dict[str, Any]`

Resolves a reference to a concept or relation in the state.

- **Parameters**:
  - `state`: Current state
  - `reference`: Reference string
- **Returns**:
  - Resolved concept, relation, or attribute

### Main Processing (`hyperlogica.py`)

#### `process_input_file(input_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`

Processes an input configuration file and generates results.

- **Parameters**:
  - `input_path`: Path to the configuration file
  - `options`: Additional processing options
- **Returns**:
  - Processing results

## Data Structures

### Concept Vector

```python
{
    "identifier": str,  # Unique identifier for the concept
    "type": str,  # Type of representation (usually "concept")
    "content": {
        "concept": str  # Core concept name
    },
    "attributes": {
        "certainty": float,  # Certainty value between 0 and 1
        # Additional domain-specific attributes
    },
    "vector": np.ndarray  # High-dimensional vector representation
}
```

### Reasoning Trace

```python
{
    "session_id": str,  # Unique session identifier
    "timestamp": str,  # ISO-8601 datetime
    "steps": [
        {
            "step_id": int,  # Sequential step ID
            "pattern": str,  # Reasoning pattern name
            "premises": List[str],  # Premise identifiers
            "conclusion": str,  # Conclusion identifier
            "certainty": float  # Certainty of the conclusion
        },
        # Additional steps...
    ],
    "final_conclusions": [
        {
            "identifier": str,  # Conclusion identifier
            "text": str,  # Natural language conclusion
            "certainty": float  # Certainty level
        },
        # Additional conclusions...
    ]
}
```

### Output Result

```python
{
    "entities_processed": int,  # Number of entities processed
    "conclusions_generated": int,  # Number of conclusions generated
    "processing_time": float,  # Processing time in seconds
    "results": [
        {
            "entity_id": str,  # Entity identifier
            "entity_name": str,  # Entity name
            "outcome": str,  # Final outcome/recommendation
            "certainty": float,  # Certainty level
            "reasoning": {
                # Domain-specific reasoning information
            }
        },
        # Additional entity results...
    ],
    "metadata": {
        "timestamp": str,  # ISO-8601 datetime
        "configuration_file": str,  # Source configuration file
        "reasoning_approach": str,  # Approach used
        "vector_dimension": int  # Dimension of vectors used
    }
}
```

## Configuration Schema

### Processing Options

```json
{
  "vector_dimension": 10000,
  "vector_type": "binary|continuous",
  "reasoning_approach": "majority|weighted|bayesian",
  "certainty_propagation": "min|product|noisy_or",
  "recalibration_enabled": true,
  "max_reasoning_depth": 10,
  "domain_config": {
    // Domain-specific configuration
  }
}
```

### Persistence Options

```json
{
  "load_previous_state": false,
  "previous_state_path": "path/to/previous/state",
  "save_state": true,
  "state_save_path": "path/to/save/state"
}
```

### Logging Options

```json
{
  "log_level": "debug|info|warning|error",
  "log_path": "path/to/logs",
  "include_vector_operations": true,
  "include_llm_interactions": true,
  "include_reasoning_steps": true
}
```

### LLM Options

```json
{
  "model": "gpt-4",
  "temperature": 0.0,
  "max_tokens": 2000
}
```

### Output Schema

```json
{
  "format": "json",
  "fields": [
    {"name": "entity_id", "type": "string"},
    {"name": "outcome", "type": "string"},
    {"name": "certainty", "type": "float"},
    {"name": "reasoning", "type": "object"}
  ],
  "include_reasoning_trace": true,
  "include_vector_details": false
}
```