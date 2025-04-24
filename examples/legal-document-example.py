html += f"""
    <div class="contract">
        <h2>{name} ({contract_type})</h2>
        <div class="details">
            <p>Risk Assessment: <span class="{risk_assessment}">{risk_assessment}</span> 
               <span class="certainty">({int(certainty * 100)}% confidence)</span></p>
            
            <div class="metrics">
                <p>Clauses: {high_risk} high risk, {standard_risk} standard risk, {low_risk} low risk</p>
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
    """Main function to run the legal document analysis example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Legal Document Analysis Example using Hyperlogica")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--input", help="Path to directory containing legal documents")
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
        
        # Process documents from input directory if provided
        if args.input:
            documents = load_documents_from_directory(args.input)
            if documents:
                # Replace the sample entities with loaded documents
                config["input_data"]["entities"] = documents
                print(f"Loaded {len(documents)} documents for analysis")
            else:
                print("No documents loaded, using sample contracts")
        
        # Process the configuration with hyperlogica
        print("Starting legal document analysis...")
        
        hyperlogica_options = {
            "verbose": args.verbose,
            "output_path": args.output if args.output else "./output/legal_analysis_results.json",
            "timeout": 600  # 10 minutes
        }
        
        results = process_input_file(
            input_path=None,  # Use in-memory configuration
            options=hyperlogica_options,
            config_dict=config  # Pass configuration directly
        )
        
        # Generate report
        report_path = args.report if args.report else "./output/legal_analysis_report.html"
        generate_report(results, report_path)
        
        print(f"Analysis complete. Processed {results['entities_processed']} documents.")
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
