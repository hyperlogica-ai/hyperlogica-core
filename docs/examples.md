# Hyperlogica Example Applications

I've created three comprehensive example applications that demonstrate how to use the Hyperlogica system in different domains. Each example follows the architecture design and showcases the flexibility of the system.

## 1. Stock Analysis Example

The `stock_analysis_example.py` script demonstrates using Hyperlogica for financial analysis of stocks, including:

- **Domain-specific configuration**: Uses a weighted reasoning approach appropriate for financial metrics
- **Sample rules**: Contains investment rules about financial indicators and their implications
- **Sample entities**: Includes three stocks with financial metrics and market data
- **HTML report generation**: Creates a formatted report of investment recommendations
- **Command-line interface**: Allows customization of input/output paths and verbosity

### Key features:
- Financial metrics analysis using vector-based reasoning
- Generation of investment recommendations (BUY/SELL/HOLD)
- Confidence levels based on weighted evaluation of factors
- Detailed explanations of the reasoning process

## 2. Legal Document Example

The `legal_document_example.py` demonstrates analyzing legal contracts with Hyperlogica:

- **Document loading**: Can load contract text from files in a directory
- **Clause extraction**: Converts document paragraphs into fact statements
- **Risk assessment**: Evaluates contracts for high, moderate, or low risk
- **Domain-specific rules**: Contains rules about contract clauses and their risk implications

### Key features:
- Document text analysis and fact extraction
- Risk evaluation of contract terms
- Identification of high-risk clauses
- Detailed explanation of contract issues and recommendations

## 3. Medical Diagnosis Example

The `medical_diagnosis_example.py` shows how Hyperlogica can support medical diagnosis:

- **Bayesian reasoning**: Uses the Bayesian approach suitable for medical diagnosis
- **Patient records**: Includes sample patients with symptoms and test results
- **Differential diagnosis**: Generates a list of possible conditions with confidence levels
- **Test recommendations**: Suggests additional tests based on differential diagnoses

### Key features:
- Symptom and test result analysis
- Diagnostic assessment with confidence levels
- Generation of differential diagnoses
- Recommended tests for further investigation
- Medical disclaimers for responsible AI use

## Common Elements Across Examples

All three examples share these common implementation patterns:

1. **Configuration management**: Creating or loading configuration from files
2. **Domain-specific rule sets**: Rules tailored to each domain's knowledge
3. **Entity processing**: Domain-specific entities (stocks, contracts, patients) with facts
4. **Report generation**: HTML output for human-readable analysis
5. **Command-line interface**: Consistent options across examples
6. **Error handling**: Comprehensive exception management

These examples demonstrate how the Hyperlogica architecture can be applied to diverse domains while maintaining consistent patterns for configuration, processing, and output. Each example is fully functional and can be extended with additional domain-specific knowledge and data sources.
