#!/usr/bin/env python3
"""
Stock Analysis Example using Hyperlogica

This example demonstrates how to use the Hyperlogica system for analyzing stocks
based on financial metrics and generating investment recommendations.

Usage:
    python stock_analysis_example.py [--config CONFIG_FILE] [--output OUTPUT_FILE] [--verbose]

The example will:
1. Load or create a configuration for stock analysis
2. Process financial metrics and investment rules
3. Apply reasoning to generate investment recommendations
4. Output detailed analysis with explanations
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the main hyperlogica module
from hyperlogica import process_input_file


# Sample stock analysis configuration
SAMPLE_STOCK_CONFIG = {
    "processing": {
        "vector_dimension": 10000,
        "vector_type": "binary",
        "reasoning_approach": "weighted",  # Use weighted approach for financial metrics
        "certainty_propagation": "min",
        "recalibration_enabled": True,
        "max_reasoning_depth": 5,
        "domain": "finance",
        "domain_config": {
            "positive_outcome_keywords": ["buy", "bullish", "upside", "growth", "undervalued"],
            "negative_outcome_keywords": ["sell", "bearish", "downside", "decline", "overvalued"],
            "neutral_outcome_keywords": ["hold", "neutral", "stable", "fair valued"],
            "outcome_field": "recommendation",
            "positive_outcome": "BUY",
            "negative_outcome": "SELL",
            "neutral_outcome": "HOLD"
        }
    },
    "persistence": {
        "load_previous_state": False,
        "save_state": True,
        "state_save_path": "./output/stock_analysis_state.pkl"
    },
    "logging": {
        "log_level": "info",
        "log_path": "./logs/stock_analysis.log",
        "include_vector_operations": False,
        "include_llm_interactions": True,
        "include_reasoning_steps": True
    },
    "llm": {
        "model": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 2000
    },
    "input_data": {
        "rules": [
            {"text": "If the P/E ratio is below industry average, then the stock is potentially undervalued.", "certainty": 0.8},
            {"text": "If revenue growth is accelerating for three consecutive quarters, then the company is in a growth phase.", "certainty": 0.9},
            {"text": "If profit margins are expanding year-over-year, then the company is improving operational efficiency.", "certainty": 0.85},
            {"text": "If a stock is undervalued and the company is in a growth phase, then it's a strong buy signal.", "certainty": 0.85},
            {"text": "If a stock is undervalued and profit margins are expanding, then it's a buy signal.", "certainty": 0.8},
            {"text": "If the debt-to-equity ratio is decreasing, then the company's financial health is improving.", "certainty": 0.75},
            {"text": "If a company has high debt and declining revenue, then it's a sell signal.", "certainty": 0.8},
            {"text": "If analyst sentiment is predominantly positive, then the stock is likely to outperform the market.", "certainty": 0.7},
            {"text": "If a stock has outperformed the market for the last 6 months, it may be overvalued.", "certainty": 0.6},
            {"text": "If a company has missed earnings expectations for two consecutive quarters, it's a cautionary signal.", "certainty": 0.75}
        ],
        "entities": [
            {
                "id": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "facts": [
                    {"text": "P/E ratio is 28.5, which is below the technology industry average of 32.8.", "certainty": 0.95},
                    {"text": "Revenue growth has accelerated for four consecutive quarters.", "certainty": 0.9},
                    {"text": "Profit margins have expanded from 21% to 23% year-over-year.", "certainty": 0.85},
                    {"text": "Debt-to-equity ratio has decreased from 1.5 to 1.2 over the past year.", "certainty": 0.9},
                    {"text": "The stock has returned 15% over the past 6 months, compared to 10% for the S&P 500.", "certainty": 0.95},
                    {"text": "85% of analysts have a positive rating on the stock.", "certainty": 0.8}
                ]
            },
            {
                "id": "MSFT",
                "name": "Microsoft Corporation",
                "sector": "Technology",
                "facts": [
                    {"text": "P/E ratio is 34.2, which is slightly above the technology industry average of 32.8.", "certainty": 0.95},
                    {"text": "Revenue growth has accelerated for three consecutive quarters.", "certainty": 0.9},
                    {"text": "Profit margins have expanded from 35% to 37% year-over-year.", "certainty": 0.85},
                    {"text": "Debt-to-equity ratio has remained stable at 0.5 over the past year.", "certainty": 0.95},
                    {"text": "The stock has returned 18% over the past 6 months, compared to 10% for the S&P 500.", "certainty": 0.95},
                    {"text": "90% of analysts have a positive rating on the stock.", "certainty": 0.85}
                ]
            },
            {
                "id": "GM",
                "name": "General Motors Company",
                "sector": "Automotive",
                "facts": [
                    {"text": "P/E ratio is 5.8, which is below the automotive industry average of 8.4.", "certainty": 0.95},
                    {"text": "Revenue growth has slowed for two consecutive quarters.", "certainty": 0.85},
                    {"text": "Profit margins have contracted from 8% to 6% year-over-year.", "certainty": 0.9},
                    {"text": "Debt-to-equity ratio has increased from 1.2 to 1.4 over the past year.", "certainty": 0.9},
                    {"text": "The stock has returned -5% over the past 6 months, compared to 10% for the S&P 500.", "certainty": 0.95},
                    {"text": "40% of analysts have a positive rating on the stock.", "certainty": 0.8}
                ]
            }
        ]
    },
    "output_schema": {
        "format": "json",
        "fields": [
            {"name": "ticker", "type": "string"},
            {"name": "name", "type": "string"},
            {"name": "sector", "type": "string"},
            {"name": "recommendation", "type": "string"},
            {"name": "certainty", "type": "float"},
            {"name": "buy_signals", "type": "integer"},
            {"name": "sell_signals", "type": "integer"},
            {"name": "reasoning", "type": "object"}
        ],
        "include_reasoning_trace": True,
        "include_explanation": True,
        "include_vector_details": False,
        "domain": "finance"
    }
}


def create_or_load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new configuration or load from file if provided.
    
    Args:
        config_path (str, optional): Path to existing configuration file
        
    Returns:
        dict: Configuration dictionary for stock analysis
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return SAMPLE_STOCK_CONFIG


def add_additional_stocks(config: Dict[str, Any], stocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add additional stocks to the configuration.
    
    Args:
        config (dict): Configuration dictionary
        stocks (list): List of stock dictionaries to add
        
    Returns:
        dict: Updated configuration dictionary
    """
    for stock in stocks:
        # Validate required fields
        if "id" not in stock or "name" not in stock or "facts" not in stock:
            logging.warning(f"Skipping invalid stock: {stock.get('id', 'unknown')}")
            continue
        
        # Add to entities
        config["input_data"]["entities"].append(stock)
    
    return config


def add_additional_rules(config: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add additional investment rules to the configuration.
    
    Args:
        config (dict): Configuration dictionary
        rules (list): List of rule dictionaries to add
        
    Returns:
        dict: Updated configuration dictionary
    """
    for rule in rules:
        # Validate required fields
        if "text" not in rule:
            logging.warning("Skipping invalid rule")
            continue
        
        # Set default certainty if not provided
        if "certainty" not in rule:
            rule["certainty"] = 0.8
        
        # Add to rules
        config["input_data"]["rules"].append(rule)
    
    return config


def generate_report(results: Dict[str, Any], output_path: Optional[str] = None) -> None:
    """
    Generate a human-readable report from the analysis results.
    
    Args:
        results (dict): Results from hyperlogica processing
        output_path (str, optional): Path where the HTML report should be saved
    """
    # Create HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Stock Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .stock {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .stock h2 {{ color: #444; margin-top: 0; }}
        .stock .details {{ margin-left: 20px; }}
        .BUY {{ color: green; font-weight: bold; }}
        .SELL {{ color: red; font-weight: bold; }}
        .HOLD {{ color: orange; font-weight: bold; }}
        .metrics {{ margin-top: 10px; }}
        .certainty {{ font-style: italic; color: #666; }}
        .explanation {{ margin-top: 15px; background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Stock Analysis Report</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Analyzed {results.get('entities_processed', 0)} stocks using {results.get('metadata', {}).get('reasoning_approach', 'unknown')} reasoning approach</p>
    
    <div class="stocks">
"""
    
    # Add each stock result
    for stock in results.get("results", []):
        ticker = stock.get("entity_id", "")
        name = stock.get("entity_name", ticker)
        sector = stock.get("sector", "Unknown")
        recommendation = stock.get("outcome", "UNKNOWN")
        certainty = stock.get("certainty", 0)
        reasoning = stock.get("reasoning", {})
        
        buy_signals = reasoning.get("positive_signals", 0)
        sell_signals = reasoning.get("negative_signals", 0)
        explanation = reasoning.get("explanation", "No explanation available.")
        
        html += f"""
    <div class="stock">
        <h2>{ticker}: {name} ({sector})</h2>
        <div class="details">
            <p>Recommendation: <span class="{recommendation}">{recommendation}</span> 
               <span class="certainty">({int(certainty * 100)}% confidence)</span></p>
            
            <div class="metrics">
                <p>Signals: {buy_signals} positive, {sell_signals} negative</p>
            </div>
            
            <div class="explanation">
                <h3>Analysis:</h3>
                <p>{explanation}</p>
            </div>
        </div>
    </div>
"""
    
    # Close HTML tags
    html += """
    </div>
</body>
</html>
"""
    
    # Save to file if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"Report saved to {output_path}")
    else:
        print(html)


def main():
    """Main function to run the stock analysis example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stock Analysis Example using Hyperlogica")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Path to save output results (JSON)")
    parser.add_argument("--report", help="Path to save HTML report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs("./output", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # Load or create configuration
        config = create_or_load_config(args.config)
        
        # Process additional example stocks (in a real scenario, these might come from a database)
        additional_stocks = []
        config = add_additional_stocks(config, additional_stocks)
        
        # Process the configuration with hyperlogica
        print("Starting stock analysis...")
        
        hyperlogica_options = {
            "verbose": args.verbose,
            "output_path": args.output if args.output else "./output/stock_analysis_results.json",
            "timeout": 600  # 10 minutes
        }
        
        results = process_input_file(
            input_path=None,  # Use in-memory configuration
            options=hyperlogica_options,
            config_dict=config  # Pass configuration directly
        )
        
        # Generate report
        report_path = args.report if args.report else "./output/stock_analysis_report.html"
        generate_report(results, report_path)
        
        print(f"Analysis complete. Processed {results['entities_processed']} stocks.")
        print(f"JSON results saved to {hyperlogica_options['output_path']}")
        print(f"HTML report saved to {report_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
