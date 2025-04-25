import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env filepi
load_dotenv()

# Verify the API key is loaded
if "OPENAI_API_KEY" in os.environ:
    print("OpenAI API key loaded successfully")
else:
    print("WARNING: OpenAI API key not found in environment variables")

# Get the project root directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(project_dir)
logs_dir = os.path.join(project_dir, "logs")
print(logs_dir)

# Create logs directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

sys.path.append(os.path.abspath('/Users/graemewoods/Development/Hyperlogica/hyperlogica-core/'))

# Import and use hyperlogica
from hyperlogica import process_input_file

# Run analysis with default configuration
results = process_input_file(
    input_path="/Users/graemewoods/Development/Hyperlogica/hyperlogica-core/examples/configs/stock_analysis_config.json",
    options={
        "verbose": True,
        "log_path": logs_dir  # Add this to specify the log path
    }
)

# Print recommendations
for result in results["results"]:
    print(f"{result['entity_id']}: {result['outcome']} ({result['certainty']:.2%} confidence)")
