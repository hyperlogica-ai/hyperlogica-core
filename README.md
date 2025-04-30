# Hyperlogica

<div align="center">

![Hyperlogica Logo](assets/logo.png)

**Vector reasoning system using hyperdimensional computing**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)

</div>

---

## 📖 Introduction

Hyperlogica is an AI reasoning system based on the AI Conceptual Exchange Protocol (ACEP), a specialized language designed for efficient and precise communication between AI systems. By leveraging hyperdimensional computing principles (10,000 dimensional vectors), Hyperlogica offers significant improvements in efficiency, precision, and reasoning capabilities.

This implementation focuses directly on ACEP representations without requiring natural language translation, providing true vector-based reasoning with explicit certainty propagation.

Hyperlogica uses high-dimensional vector representations with explicit relationship markers, certainty qualifiers, and a state reference system that dramatically reduces token requirements and computational overhead.


---

## ✨ Key features

- **Direct ACEP input processing**: Works with structured ACEP representations using proper concept binding
- **Hyperdimensional vector operations**: Implements binding, unbinding, and bundling for semantic representation
- **Vector-chain reasoning**: Creates explicit reasoning chains with controlled uncertainty propagation
- **Automatic explanation generation**: Provides human-readable explanations of reasoning processes
- **FAISS integration**: Uses FAISS for efficient vector storage and retrieval

---

## 🚀 Getting started

### Prerequisites

- Python 3.8 or higher
- FAISS for vector storage and retrieval
- NumPy for vector operations

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hyperlogica/hyperlogica.git
cd hyperlogica
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script:
```bash
pip install -e .
```

### Quick example

```python
from hyperlogica import process_input_file

# Run analysis with ACEP configuration
results = process_input_file(
    input_path="examples/stock_example_config.json",
    options={"verbose": True}
)

# Print recommendations
for result in results["results"]:
    print(f"{result['entity_id']}: {result['outcome']} ({result['certainty']:.2%} confidence)")
```

---

## 📊 ACEP representation

Hyperlogica uses a structured ACEP format for both rules and facts:

### Rules (Conditional relations)

```json
{
  "identifier": "rule_pe_below_industry",
  "type": "conditional_relation",
  "content": {
    "condition": {
      "concept": "pe_ratio",
      "relation": "below",
      "reference": "industry_average"
    },
    "implication": {
      "concept": "valuation",
      "state": "undervalued"
    }
  },
  "attributes": {
    "certainty": 0.8,
    "domain": "finance",
    "source": "financial_analysis"
  }
}
```

### Facts (Factual assertions)

```json
{
  "identifier": "AAPL_pe_ratio",
  "type": "factual_assertion",
  "content": {
    "concept": "pe_ratio",
    "relation": "below",
    "reference": "industry_average",
    "actual_value": 28.5,
    "reference_value": 32.8
  },
  "attributes": {
    "certainty": 0.95,
    "source": "financial_statements",
    "timestamp": "2023-12-31"
  }
}
```

---

## 🧠 Vector-chain reasoning

Hyperlogica uses a structured vector-chain reasoning approach:

1. Generate vectors for rule conditions, implications, and facts
2. Match facts to rule conditions using vector similarity
3. Create conclusion vectors and track certainty
4. Build chains of reasoning for multi-step inference
5. Classify conclusions as positive, negative, or neutral
6. Produce final recommendations with confidence levels

---

## 🛠️ Project structure

```
hyperlogica/
│
├── hyperlogica/             # Core package
│   ├── __init__.py          # Package initialization
│   ├── config_parser.py     # Configuration handling
│   ├── vector_operations.py # Vector mathematical operations
│   ├── vector_store.py      # Vector storage and retrieval
│   ├── reasoning_engine.py  # Reasoning implementation
│   ├── state_management.py  # Context and state management
│   ├── error_handling.py    # Error handling utilities
│   └── hyperlogica.py       # Main entry point
│
├── tests/                   # Test suite
│   ├── test_vector_operations.py
│   └── test_reasoning_engine.py
│
├── examples/                # Example configurations
│   └── stock_example_config.json
│
├── README.md                # This file
├── setup.py                 # Package installation script
└── requirements.txt         # Package dependencies
```

---

## 🧪 Running tests

To run the test suite:

```bash
pytest tests/
```

---

## 🐳 Docker

You can build and run Hyperlogica using Docker:

```bash
docker build -t hyperlogica .
docker run -v $(pwd)/examples:/app/examples -v $(pwd)/output:/app/output hyperlogica python -m hyperlogica.hyperlogica examples/stock_example_config.json
```

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
