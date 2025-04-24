# example_llm_interface.py
"""
Example usage of Hyperlogica LLM Interface

This script demonstrates how to use the LLM interface module to:
1. Convert natural language text to ACEP representation
2. Convert ACEP representation back to natural language
3. Generate explanations from reasoning traces
4. Use caching to optimize API calls
5. Work with vector embeddings

To run this example:
1. Set the OPENAI_API_KEY environment variable
2. Run: python example_llm_interface.py
"""

import os
import json
import time
import numpy as np
import llm_interface

def stock_analysis_example():
    """
    Demonstrate using the LLM interface for stock analysis.
    """
    print("\n=== Stock Analysis Example ===\n")
    
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Example text to convert - a simple rule
    rule_text = "If a company's P/E ratio is below the industry average, then the stock might be undervalued."
    
    # Set up context for finance domain
    context = {
        "domain": "finance",
        "entity_id": "AAPL",
        "additional_context": "Stock analysis for a technology company"
    }
    
    # Convert rule to ACEP representation
    try:
        print(f"Converting rule to ACEP: {rule_text}")
        rule_acep = llm_interface.convert_english_to_acep(rule_text, context)
        print(f"\nACEP representation:")
        print(json.dumps(rule_acep, indent=2))
        
        # Fact to analyze
        fact_text = "Apple's P/E ratio of 28.5 is below the technology industry average of 32.8."
        
        # Convert fact to ACEP
        print(f"\nConverting fact to ACEP: {fact_text}")
        fact_acep = llm_interface.convert_english_to_acep(fact_text, context)
        print(f"\nACEP representation:")
        print(json.dumps(fact_acep, indent=2))
        
        # Create a reasoning trace
        print("\nCreating reasoning trace with modus ponens...")
        reasoning_trace = {
            "session_id": "example_session",
            "timestamp": "2023-04-15T10:30:00Z",
            "steps": [
                {
                    "step_id": 1,
                    "pattern": "modus_ponens",
                    "premises": [
                        rule_acep["identifier"],
                        fact_acep["identifier"]
                    ],
                    "conclusion": "apple_stock_undervalued",
                    "certainty": min(rule_acep["attributes"]["certainty"], fact_acep["attributes"]["certainty"])
                }
            ],
            "final_conclusions": [
                {
                    "identifier": "apple_stock_undervalued",
                    "text": "Apple stock is potentially undervalued",
                    "certainty": min(rule_acep["attributes"]["certainty"], fact_acep["attributes"]["certainty"])
                }
            ]
        }
        
        # Generate explanation
        explanation_context = {
            "domain": "finance",
            "entity_id": "AAPL",
            "recommendation": "CONSIDER_BUY",
            "certainty": min(rule_acep["attributes"]["certainty"], fact_acep["attributes"]["certainty"])
        }
        
        print("\nGenerating explanation from reasoning trace...")
        explanation = llm_interface.generate_explanation(reasoning_trace, explanation_context)
        print("\nGenerated explanation:")
        print(explanation)
        
    except Exception as e:
        print(f"Error: {str(e)}")

def caching_example():
    """
    Demonstrate using API call caching for efficiency.
    """
    print("\n=== Caching Example ===\n")
    
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Example prompt
    prompt = "Summarize the advantages of vector-based representation in AI communication."
    
    try:
        # First call (should hit the API)
        print("Making first API call...")
        start_time = time.time()
        response1 = llm_interface.call_openai_api_cached(
            prompt=prompt, 
            model="gpt-3.5-turbo", 
            cache_file="example_cache.json"
        )
        elapsed1 = time.time() - start_time
        print(f"First call completed in {elapsed1:.2f} seconds")
        
        # Second call (should use cache)
        print("\nMaking second API call (should use cache)...")
        start_time = time.time()
        response2 = llm_interface.call_openai_api_cached(
            prompt=prompt, 
            model="gpt-3.5-turbo", 
            cache_file="example_cache.json"
        )
        elapsed2 = time.time() - start_time
        print(f"Second call completed in {elapsed2:.2f} seconds")
        
        # Compare times
        print(f"\nTime comparison: First call: {elapsed1:.2f}s, Second call: {elapsed2:.2f}s")
        print(f"Cache speedup: {elapsed1/elapsed2:.1f}x faster")
        
        # Check if responses match
        if response1["choices"][0]["message"]["content"] == response2["choices"][0]["message"]["content"]:
            print("Responses match! Caching is working correctly.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def embedding_example():
    """
    Demonstrate working with vector embeddings.
    """
    print("\n=== Embedding Example ===\n")
    
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Create embeddings for related and unrelated concepts
        print("Creating embeddings for concepts...")
        
        # Related financial concepts
        concept1 = "A company's P/E ratio being below industry average"
        concept2 = "The stock being undervalued based on earnings multiples"
        
        # Unrelated concept
        concept3 = "The weather forecast predicts rain tomorrow"
        
        # Get embeddings
        print(f"\nGenerating embedding for: '{concept1}'")
        embedding1, metadata1 = llm_interface.create_embedding(concept1)
        
        print(f"Generating embedding for: '{concept2}'")
        embedding2, metadata2 = llm_interface.create_embedding(concept2)
        
        print(f"Generating embedding for: '{concept3}'")
        embedding3, metadata3 = llm_interface.create_embedding(concept3)
        
        # Calculate similarities
        similarity12 = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity13 = np.dot(embedding1, embedding3) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding3))
        similarity23 = np.dot(embedding2, embedding3) / (np.linalg.norm(embedding2) * np.linalg.norm(embedding3))
        
        print(f"\nEmbedding dimensions: {metadata1['dimensions']}")
        print(f"\nSimilarity between related concepts (1 & 2): {similarity12:.4f}")
        print(f"Similarity between unrelated concepts (1 & 3): {similarity13:.4f}")
        print(f"Similarity between unrelated concepts (2 & 3): {similarity23:.4f}")
        
        # Demonstrate deterministic vectors as fallback
        print("\nDemonstrating deterministic vectors (as fallback when API is unavailable):")
        det_vec1 = llm_interface.generate_deterministic_vector(concept1, 1536)
        det_vec2 = llm_interface.generate_deterministic_vector(concept2, 1536)
        det_vec3 = llm_interface.generate_deterministic_vector(concept3, 1536)
        
        det_sim12 = np.dot(det_vec1, det_vec2) / (np.linalg.norm(det_vec1) * np.linalg.norm(det_vec2))
        det_sim13 = np.dot(det_vec1, det_vec3) / (np.linalg.norm(det_vec1) * np.linalg.norm(det_vec3))
        
        print(f"Deterministic similarity (1 & 2): {det_sim12:.4f}")
        print(f"Deterministic similarity (1 & 3): {det_sim13:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def utility_functions_example():
    """
    Demonstrate the utility functions for text processing.
    """
    print("\n=== Utility Functions Example ===\n")
    
    # Normalized identifiers
    print("Creating normalized identifiers:")
    examples = [
        "If P/E ratio is low, then stock is undervalued.",
        "Revenue growth has accelerated for 3 consecutive quarters!",
        "The company's debt-to-equity ratio is 0.45 (below industry average)"
    ]
    
    for text in examples:
        identifier = llm_interface.create_normalized_identifier(text)
        print(f"  Original: '{text}'")
        print(f"  Identifier: '{identifier}'")
        print()
    
    # Certainty extraction
    print("\nExtracting certainty from language:")
    certainty_examples = [
        "The stock will definitely rise after the earnings report.",
        "The company will very likely beat analyst expectations.",
        "The stock price will probably increase in the short term.",
        "The market might respond positively to the announcement.",
        "It is unlikely that the acquisition will be approved.",
        "It is almost certainly not going to meet revenue targets."
    ]
    
    for text in certainty_examples:
        certainty = llm_interface.extract_certainty_language(text)
        print(f"  '{text}'")
        print(f"  Certainty: {certainty:.2f}")
        print()
    
    # Detecting conditional statements
    print("\nDetecting conditional statements:")
    conditional_examples = [
        "If the market rises, then tech stocks will follow.",
        "When interest rates fall, bond prices increase.",
        "The stock price has been stable for the past month.",
        "Whenever a company announces layoffs, the stock typically drops.",
        "The quarterly report shows strong revenue growth."
    ]
    
    for text in conditional_examples:
        is_conditional = llm_interface.is_conditional_statement(text)
        print(f"  '{text}'")
        print(f"  Is conditional: {is_conditional}")
        print()
    
    # Extracting numeric values
    print("\nExtracting numeric values:")
    numeric_examples = [
        "The company reported 15% revenue growth.",
        "The P/E ratio is 22.5, above the industry average.",
        "Market share increased by 2.7 percentage points.",
        "No specific numbers mentioned in this analysis."
    ]
    
    for text in numeric_examples:
        value = llm_interface.extract_numeric_value(text)
        print(f"  '{text}'")
        print(f"  Extracted value: {value}")
        print()
    
    # Extracting temporal references
    print("\nExtracting temporal references:")
    temporal_examples = [
        "The earnings report will be released tomorrow.",
        "The stock has increased by 10% over the last week.",
        "The company plans to launch the product in 2023-07-15.",
        "Long-term prospects remain strong for the sector.",
        "The dividend will be paid in 30 days."
    ]
    
    for text in temporal_examples:
        temporal = llm_interface.extract_temporal_reference(text)
        print(f"  '{text}'")
        print(f"  Temporal reference: {temporal}")
        print()

def main():
    """Main function to run all examples."""
    print("\n==================================================")
    print("    HYPERLOGICA LLM INTERFACE - EXAMPLES")
    print("==================================================\n")
    
    # Display info about API key
    if "OPENAI_API_KEY" in os.environ:
        print("OpenAI API key found. Examples will attempt to make API calls.")
    else:
        print("WARNING: OpenAI API key not found. Set OPENAI_API_KEY environment variable to run the examples.")
        return
    
    # Run the examples
    stock_analysis_example()
    caching_example()
    embedding_example()
    utility_functions_example()
    
    print("\n==================================================")
    print("             EXAMPLES COMPLETED")
    print("==================================================\n")

if __name__ == "__main__":
    main()