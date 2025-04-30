from hyperlogica import process_input_file

# Run analysis with ACEP configuration
results = process_input_file(
    input_path="/Users/graemewoods/Development/Hyperlogica/hyperlogica-core/examples/stock_example_config.json",
    options={"verbose": True}
)

# Print recommendations
for result in results["results"]:
    print(f"{result['entity_id']}: {result['outcome']} ({result['certainty']:.2%} confidence)")