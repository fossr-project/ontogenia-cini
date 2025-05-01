import requests

VALIDATOR_URL = "http://127.0.0.1:8000/validate/"
GENERATOR_URL = "http://127.0.0.1:8001/newapi/"

payload = {
    # Ask the validator to use its own default dataset
    "use_default_dataset": "true",
    # Point it at your live CQ-generator
    "external_service_url": GENERATOR_URL,
    # If your generator needs an API key, include it here:
    "api_key": "sk-...",
    # Which validations to run
    "validation_mode": "all",
    # LLM model for the LLM-based checks
    "model": "gpt-4",
    # Where it should dump heatmaps
    "output_folder": "heatmaps",
    # Whether to save a CSV of results server-side
    "save_results": "true",
}

resp = requests.post(VALIDATOR_URL, data=payload)
print("Status:", resp.status_code)
print(resp.json())
