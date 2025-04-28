#!/usr/bin/env python3
"""
Hyperlogica Ontology-Enhanced Stock Analysis Example

This script demonstrates how to use the ontology-enhanced Hyperlogica system
for stock analysis with improved term standardization.
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify the API key is loaded
if "OPENAI_API_KEY" in os.environ:
    print("OpenAI API key loaded successfully")
else:
    print("WARNING: OpenAI API key not found in environment variables")

# Add parent directory to path if necessary
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the enhanced hyperlogica modules
from hyperlogica import process_input_file
from hyperlogica.ontology_mapper import create_ontology_mapper, map_text_to_ontology


def run_stock_analysis_with_ontology(config_path: str, output_path: str = None, verbose: bool = False):
    """
    Run stock analysis with ontology-enhanced processing.
    
    Args:
        config_path: Path to the configuration file
        output_path: Path to save the results
        verbose: Whether to enable verbose output
    """
    # Ensure config_path is an absolute path
    if not os.path.isabs(config_path):
        # First check if the path is relative to the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, config_path)
        
        if os.path.exists(potential_path):
            config_path = potential_path
        else:
            # Try relative to project root
            project_root = os.path.abspath(os.path.join(script_dir, "../.."))
            config_path = os.path.join(project_root, config_path)
    
    # Verify the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Set up options
    options = {
        "verbose": verbose,
        "output_path": output_path or "./output/stock_analysis_results.json"
    }
    
    print(f"Processing configuration from {config_path}...")
    
    # Run the analysis
    results = process_input_file(
        input_path=config_path,
        options=options
    )
    
    # Print recommendations
    print("\nStock Analysis Results:")
    print(f"Processed {len(results['results'])} stocks in {results['processing_time']:.2f} seconds")
    print("-" * 50)
    
    for result in results["results"]:
        print(f"{result['entity_id']}: {result['outcome']} ({result['certainty']:.2%} confidence)")
    
    return results


def demonstrate_ontology_mapping(config_path: str):
    """
    Demonstrate how the ontology mapper works with example phrases.
    
    Args:
        config_path: Path to the configuration file with ontology
    """
    # Ensure config_path is an absolute path
    if not os.path.isabs(config_path):
        # First check if the path is relative to the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, config_path)
        
        if os.path.exists(potential_path):
            config_path = potential_path
        else:
            # Try relative to project root
            project_root = os.path.abspath(os.path.join(script_dir, "../.."))
            config_path = os.path.join(project_root, config_path)
    
    # Verify the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the configuration with ontology
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract domain configuration with ontology
    processing_options = config.get("processing", {})
    domain_config = processing_options.get("domain_config", {})
    
    # Create the ontology mapper
    mapper = create_ontology_mapper(domain_config)
    
    # Example phrases to test
    test_phrases = [
        "P/E ratio is 15.2, which is below the industry average of 18.7",
        "Revenue growth has accelerated for the last three consecutive quarters",
        "Profit margins have expanded from 12% to 15% year-over-year",
        "The debt-to-equity ratio has decreased from 2.1 to 1.8",
        "The stock has returned 20% over the past 6 months, outperforming the S&P 500",
        "85% of analysts have a buy rating on the stock"
    ]
    
    print("\nOntology Mapping Examples:")
    print("-" * 50)
    
    finance_ontology = domain_config.get("finance_ontology", {})
    
    for phrase in test_phrases:
        term, confidence = map_text_to_ontology(phrase, mapper)
        
        if term:
            print(f"Phrase: \"{phrase}\"")
            print(f"Mapped to: {term} (confidence: {confidence:.2f})")
            
            # Find category for this term - safely traverse the finance ontology structure
            category = None
            for cat, terms in finance_ontology.items():
                if isinstance(terms, dict) and term in terms:
                    category = cat
                    break
            
            if category:
                print(f"Category: {category}")
            
            print()
        else:
            print(f"Phrase: \"{phrase}\"")
            print("No mapping found")
            print()


def main():
    """Main function."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Hyperlogica Ontology-Enhanced Stock Analysis")
    parser.add_argument("--config", default="examples/configs/stock_analysis_config.json", 
                      help="Path to configuration file")
    parser.add_argument("--output", help="Path to save output results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--demo-only", action="store_true", 
                      help="Only demonstrate ontology mapping without full analysis")
    
    args = parser.parse_args()
    
    try:
        # Create output directory if needed
        os.makedirs("./output", exist_ok=True)
        
        if args.demo_only:
            # Just demonstrate the ontology mapping
            demonstrate_ontology_mapping(args.config)
        else:
            # Run the full analysis
            results = run_stock_analysis_with_ontology(
                config_path=args.config,
                output_path=args.output,
                verbose=args.verbose
            )
            
            # Also demonstrate the ontology mapping
            demonstrate_ontology_mapping(args.config)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
