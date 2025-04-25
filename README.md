# Hyperlogica

<div align="center">

![Hyperlogica Logo](assets/logo.png)

**Advanced AI reasoning system based on hyperdimensional computing**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/API-OpenAI-83BD75.svg)](https://openai.com/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)

</div>

---

## 📖 Introduction

Hyperlogica is a sophisticated AI communication system that implements the AI Conceptual Exchange Protocol (ACEP), a specialized language designed for efficient and precise communication between AI systems. By leveraging hyperdimensional computing principles, Hyperlogica overcomes the limitations of natural language for machine-to-machine communication, offering significant improvements in efficiency, precision, and reasoning capabilities.

Unlike natural language, which is inherently ambiguous and inefficient for AI communication, Hyperlogica uses high-dimensional vector representations with explicit relationship markers, certainty qualifiers, and a state reference system that dramatically reduces token requirements and computational overhead.

---

## ✨ Key Features

- **Vector-Based Conceptual Tokens** - Encode meaning in 10,000+ dimensional vector space for precise semantic representation
- **Explicit Relationship Markers** - Define clear causal, conditional, and hierarchical connections between concepts
- **State Reference System** - Efficiently manage context without repetitive information exchange
- **Parallel Information Streams** - Process multiple data streams simultaneously rather than sequentially
- **Embedded Certainty and Metadata** - Include explicit probability assessments and computational requirements
- **Bounded Syllogistic Reasoning** - Implement controlled reasoning chains with precise uncertainty propagation
- **Interoperability with LLMs** - Translate between natural language and ACEP representations

---

## 🏗️ System Architecture

Hyperlogica follows a modular architecture designed for flexibility, extensibility, and performance:

```
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│  Input JSON     │   │                  │   │  Output JSON    │
│  Configuration  │──▶│  Hyperlogica     │──▶│  Results        │
│  and Data       │   │  Processing      │   │                 │
└─────────────────┘   └──────────────────┘   └─────────────────┘
                               ▲
                               │
                      ┌────────┴───────┐
                      │                │
                ┌─────┴─────┐    ┌─────┴─────┐
                │   FAISS   │    │  OpenAI   │
                │  Vector   │    │    API    │
                │   Store   │    │           │
                └───────────┘    └───────────┘
```

### Core Components

1. **Hyperdimensional Vector Engine (HVE)**
   - Manages 10,000+ dimensional vector representations
   - Implements vector operations (binding, bundling, permutation)
   - Provides optimized similarity search capabilities

2. **Concept Vector Store (CVS)**
   - Maintains persistent storage of concept vectors
   - Manages relationships between concepts
   - Provides versioning and lineage tracking

3. **Syllogistic Reasoning System (SRS)**
   - Implements bounded reasoning structures
   - Manages uncertainty propagation
   - Validates logical consistency

4. **Knowledge Acquisition Manager (KAM)**
   - Handles ingestion from multiple data sources
   - Reconciles conflicting information
   - Manages continuous learning

5. **LLM Integration Layer (LIL)**
   - Interfaces with external language models
   - Translates between natural language and vector representations
   - Provides fallback capabilities

6. **State Management System (SMS)**
   - Maintains communication session state
   - Implements reference resolution
   - Manages context windows efficiently

7. **Domain-Specific Adapters (DSA)**
   - Specialized modules for business domains
   - Custom reasoning patterns for specific use cases
   - Domain-specific knowledge structures

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- FAISS for vector storage and retrieval
- OpenAI API key for LLM integration
- NumPy for vector operations
- Recommended: GPU support for faster vector operations

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hyperlogica/hyperlogica-core/hyperlogica.git
cd hyperlogica
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

4. Run the setup script:
```bash
python setup.py install
```

### Quick Example

```python
from hyperlogica import process_input_file

# Run analysis with default configuration
results = process_input_file(
    input_path="examples/configs/stock_analysis_config.json",
    options={"verbose": True}
)

# Print recommendations
for result in results["results"]:
    print(f"{result['entity_id']}: {result['outcome']} ({result['certainty']:.2%} confidence)")
```

---

## 📊 Example Applications

### 1. Stock Analysis System

A comprehensive stock analysis tool that evaluates financial metrics and generates investment recommendations.

**Features:**
- Rule-based analysis of financial indicators
- Weighted evidence reasoning for investment decisions
- Certainty-aware recommendation generation
- Detailed explanation of investment logic

**Running the example:**
```bash
python examples/stock_analysis_example.py [--config CONFIG_FILE] [--output OUTPUT_FILE] [--report REPORT_FILE] [--verbose]
```

**Example configuration:**
```json
{
  "processing": {
    "vector_dimension": 10000,
    "reasoning_approach": "weighted",
    "domain_config": {
      "positive_outcome": "BUY",
      "negative_outcome": "SELL",
      "neutral_outcome": "HOLD"
    }
  },
  "input_data": {
    "rules": [
      {"text": "If the P/E ratio is below industry average, then the stock is potentially undervalued.", "certainty": 0.8},
      {"text": "If revenue growth is accelerating, then the company is in a growth phase.", "certainty": 0.9}
    ],
    "entities": [
      {
        "id": "AAPL",
        "name": "Apple Inc.",
        "facts": [
          {"text": "P/E ratio is 28.5, which is below the industry average of 32.8.", "certainty": 0.95},
          {"text": "Revenue growth has accelerated for four consecutive quarters.", "certainty": 0.9}
        ]
      }
    ]
  }
}
```

### 2. Legal Document Analyzer

Analyzes legal contracts and documents to identify risks, obligations, and potential issues.

**Features:**
- Clause-level risk assessment
- Identification of high-risk contractual terms
- Comparison against standard templates
- Risk rating with confidence levels

**Running the example:**
```bash
python examples/legal_document_example.py [--config CONFIG_FILE] [--input DOCUMENT_DIR] [--output OUTPUT_FILE] [--report REPORT_FILE]
```

### 3. LLM Interface Explorer

Demonstrates the LLM interface functionality for translating between natural language and ACEP representations.

**Features:**
- English to ACEP conversion with vector generation
- ACEP to English translation
- Reasoning trace explanation generation
- API caching demonstration
- Vector similarity analysis

**Running the example:**
```bash
python examples/example_llm_interface.py
```

---

## 🧩 JSON Format Specifications

Hyperlogica uses structured JSON formats throughout its pipeline:

### Configuration Schema

```json
{
  "processing": {
    "vector_dimension": 10000,
    "vector_type": "binary|continuous",
    "reasoning_approach": "majority|weighted|bayesian",
    "certainty_propagation": "min|product|noisy_or",
    "recalibration_enabled": true,
    "max_reasoning_depth": 10,
    "domain_config": {
      // Domain-specific configuration parameters
      "positive_outcome_keywords": ["positive", "increase", "growth"],
      "negative_outcome_keywords": ["negative", "decrease", "decline"],
      "outcome_field": "recommendation"
    }
  },
  "persistence": {
    "load_previous_state": false,
    "previous_state_path": "path/to/previous/state",
    "save_state": true,
    "state_save_path": "path/to/save/state"
  },
  "logging": {
    "log_level": "debug|info|warning|error",
    "log_path": "path/to/logs",
    "include_vector_operations": true,
    "include_llm_interactions": true
  },
  "llm": {
    "model": "gpt-4",
    "temperature": 0.0,
    "max_tokens": 2000
  },
  "input_data": {
    "rules": [
      {"text": "Rule statement", "certainty": 0.9}
    ],
    "entities": [
      {
        "id": "entity_id",
        "name": "Entity Name",
        "facts": [
          {"text": "Factual statement", "certainty": 0.95}
        ]
      }
    ]
  },
  "output_schema": {
    "format": "json",
    "fields": [
      {"name": "field_name", "type": "field_type"}
    ],
    "include_reasoning_trace": true
  }
}
```

### Vector Representation

```python
{
    "identifier": "string",
    "vector": np.ndarray,  # or binary array
    "metadata": {
        "source": "rule|fact|derived",
        "text": "original text",
        "certainty": float,
        "attributes": {  # domain-specific attributes 
            "key1": "value1",
            "key2": "value2"
        }
    }
}
```

### ACEP Representation

```python
{
    "type": "concept|relation|operation",
    "identifier": "string",
    "content": {
        "concept": "string",  # for concept type
        "relation_type": "string",  # for relation type
        "source": "identifier",  # for relation type
        "target": "identifier",  # for relation type
        "operation_type": "string",  # for operation type
        "parameters": []  # for operation type
    },
    "attributes": {
        "temporal": "value",
        "certainty": float,
        # Additional attributes...
    },
    "vector": np.ndarray  # or binary array
}
```

### Reasoning Trace

```python
{
    "session_id": "string",
    "timestamp": "ISO-8601 datetime",
    "steps": [
        {
            "step_id": 1,
            "pattern": "modus_ponens",
            "premises": ["identifier1", "identifier2"],
            "conclusion": "identifier3",
            "certainty": float,
            "vector_operations": []  # detailed operations if requested
        }
    ],
    "final_conclusions": [
        {
            "identifier": "string",
            "text": "natural language conclusion",
            "certainty": float,
            "derivation_path": [step_id, step_id, ...]
        }
    ]
}
```

---

## 📂 Project Structure

```
hyperlogica-core/
│
├── config/                       # Configuration files
│   ├── default_config.json       # Default configuration
│   ├── domain_configs/           # Domain-specific configurations
│   └── logging_config.json       # Logging configuration
│
├── hyperlogica/                  # Core package
│   ├── __init__.py               # Package initialization
│   ├── config_parser.py          # Configuration handling
│   ├── vector_operations.py      # Vector mathematical operations
│   ├── vector_store.py           # Vector storage and retrieval
│   ├── llm_interface.py          # Language model integration
│   ├── reasoning_engine.py       # Reasoning implementation
│   ├── reasoning_approaches.py   # Different reasoning strategies
│   ├── state_management.py       # Context and state management
│   ├── error_handling.py         # Error handling utilities
│   ├── logging_utils.py          # Logging functions
│   └── extensions.py             # Extension registration
│
├── examples/                     # Example applications
│   ├── configs/                  # Example configurations
│   ├── stock_analysis_example.py # Stock analysis example
│   ├── legal_document_example.py # Legal document analysis example
│   └── example_llm_interface.py  # LLM interface example
│
├── tests/                        # Test suite
│   ├── test_vector_operations.py # Vector operation tests
│   ├── test_reasoning_engine.py  # Reasoning engine tests
│   ├── test_llm_interface.py     # LLM interface tests
│   └── test_end_to_end.py        # End-to-end system tests
│
├── docs/                         # Documentation
│   ├── architecture.md           # Architecture documentation
│   ├── api_reference.md          # API reference
│   └── examples.md               # Example documentation
│
├── output/                       # Default output directory
│   └── .gitkeep                  # Placeholder for git
│
├── README.md                     # This file
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # Apache 2.0 License
├── setup.py                      # Package installation script
└── requirements.txt              # Package dependencies
```

---

## 🧠 Technical Details

### Vector Operations

Hyperlogica implements these core hyperdimensional computing operations:

1. **Binding** - Associates two concepts in a relationship
   ```python
   # Binary vectors (XOR operation)
   def bind_binary(vector_a, vector_b):
       return np.logical_xor(vector_a, vector_b).astype(np.int8)
   
   # Real-valued vectors (circular convolution)
   def bind_continuous(vector_a, vector_b):
       return np.real(np.fft.ifft(np.fft.fft(vector_a) * np.fft.fft(vector_b)))
   ```

2. **Bundling** - Combines multiple related concepts
   ```python
   def bundle_vectors(vectors, weights=None):
       if weights is None:
           weights = np.ones(len(vectors)) / len(vectors)
       result = np.zeros_like(vectors[0], dtype=float)
       for i, vector in enumerate(vectors):
           result += weights[i] * vector
       return result / np.linalg.norm(result)
   ```

3. **Permutation** - Encodes order information
   ```python
   def permute_vector(vector, shift):
       return np.roll(vector, shift)
   ```

### Reasoning Patterns

Hyperlogica implements these reasoning patterns:

1. **Modus Ponens** - "If P → Q and P, then Q"
   - Example: "If low P/E then undervalued; P/E is low; therefore stock is undervalued"
   - Certainty: min(certainty(P→Q), certainty(P))

2. **Conjunction Introduction** - "P, Q, therefore P∧Q"
   - Example: "Stock is undervalued; Company is growing; therefore stock is undervalued and growing"
   - Certainty: min(certainty(P), certainty(Q))

3. **Hypothetical Syllogism** - "If P→Q and Q→R, then P→R"
   - Example: "If undervalued then buy; If growing then undervalued; therefore if growing then buy"
   - Certainty: min(certainty(P→Q), certainty(Q→R))

4. **Bounded Reasoning** - Chains of syllogisms with controlled uncertainty propagation
   - Prevents error accumulation in long reasoning chains
   - Allows explicit recalibration between reasoning steps

### Performance Considerations

For optimal performance:

- Use binary vectors (`vector_type: "binary"`) for larger vector stores
- Implement FAISS with appropriate index type (HNSW for large stores)
- Enable API caching for repeated LLM calls
- Use parallel processing for independent entities
- Consider dimensionality reduction for edge deployments

---

## 🛠️ Advanced Usage

### Custom Reasoning Approaches

You can register custom reasoning approaches:

```python
from hyperlogica.extensions import register_reasoning_approach

@register_reasoning_approach("custom_approach")
def custom_reasoning_approach(rules, facts, store, state, config):
    # Custom reasoning implementation
    # Must return a dictionary with at least "outcome" and "certainty" keys
    return {
        "outcome": "CUSTOM_OUTCOME",
        "certainty": 0.85,
        "custom_data": {...}
    }
```

### Vector Store Configuration

For larger datasets, use optimized FAISS configurations:

```json
"vector_store": {
  "type": "faiss",
  "index_type": "hnsw",
  "hnsw_params": {
    "M": 16,
    "efConstruction": 200,
    "efSearch": 128
  },
  "dimension": 10000,
  "metric": "ip"  // inner product
}
```

### Custom Domain Adapters

Create specialized domain adapters:

```python
from hyperlogica.extensions import register_domain_adapter

@register_domain_adapter("finance")
class FinanceAdapter:
    def __init__(self, config):
        self.config = config
        # Domain-specific initialization
        
    def process_entity(self, entity):
        # Domain-specific processing
        return processed_entity
        
    def interpret_result(self, result):
        # Domain-specific interpretation
        return interpreted_result
```

---

## 📈 Benefits and Applications

Hyperlogica offers significant advantages for AI-to-AI communication:

| Benefit | Description |
|---------|-------------|
| **Token Efficiency** | Represents complex ideas with 50-70% fewer tokens than natural language |
| **Precision** | Explicit encoding of relationship types and certainty values |
| **Computational Alignment** | Direct mapping to AI vector representations without translation loss |
| **Uncertainty Handling** | Precise propagation of probability values through reasoning chains |
| **Context Management** | Efficient state references eliminate redundant context rebuilding |
| **Parallel Processing** | Multiple information streams processed simultaneously |

### Ideal Application Areas

- **Business Automation** - Rule processing with certainty propagation
- **Financial Analysis** - Complex investment reasoning with risk assessment
- **Legal Document Processing** - Contract analysis and risk identification
- **Cross-Domain Knowledge Transfer** - Knowledge translation between specialized domains
- **Multi-Agent Systems** - Efficient communication between specialized AI agents
- **Edge Computing** - Lightweight reasoning on resource-constrained devices

---

## 📝 License

This project is licensed under the [Apache License 2.0](LICENSE) - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 References

- [Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors](https://doi.org/10.1007/s12559-009-9009-8)
- [Vector Symbolic Architectures: A New Building Block for Artificial General Intelligence](https://agi-conference.org/2010/wp-content/uploads/2009/06/paper_15.pdf)
- [Conceptual Exchange Protocol (ACEP) Specification v1.2](docs/acep_specification.md)
- [Reasoning with Bounded Syllogisms](docs/bounded_syllogisms.md)

---

<div align="center">
  <br>
  <p>Built with ❤️ by the Hyperlogica Team</p>
  <p>© 2025 Hyperlogica Project</p>
</div>
