import io
import json
from fastapi.testclient import TestClient
from cqver_rest_api import app

client = TestClient(app)

API_KEY = "sk-key"

# ----- Test Generation Endpoint -----

csv_data = "col1,col2,scenario\nval1,val2,Test Scenario\nval3,val4,Test Scenario\n"
file_bytes = csv_data.encode("utf-8")

# 1. Test generation with "dataset+description" mode.
response = client.post(
    "/generate",
    data={
        "dataset_description": "Real dataset description",
        "generation_mode": "dataset+description",
        "dataset_sample": "Real dataset sample provided.",
        "api_key": API_KEY,
        "model": "gpt-4"
    },
    files={"file": ("real.csv", file_bytes, "text/csv")}
)
print("Generation (dataset+description) response:")
print(json.dumps(response.json(), indent=2), "\n")

# 2. Test generation with "user_stories" mode.
response = client.post(
    "/generate",
    data={
        "generation_mode": "user_stories",
        "user_stories": "Real user story 1. Real user story 2.",
        "api_key": API_KEY,
        "model": "gpt-4"
    },
    files={"file": ("real.csv", file_bytes, "text/csv")}
)
print("Generation (user_stories) response:")
print(json.dumps(response.json(), indent=2), "\n")

# 3. Test generation with "dataset" mode.
response = client.post(
    "/generate",
    data={
        "generation_mode": "dataset",
        "dataset_sample": "Real custom dataset sample only.",
        "api_key": API_KEY,
        "model": "gpt-4"
    },
    files={"file": ("real.csv", file_bytes, "text/csv")}
)
print("Generation (dataset) response:")
print(json.dumps(response.json(), indent=2), "\n")

# ----- Test Validation Endpoint -----

# Prepare CSV data for validation (must contain both 'gold standard' and 'generated' columns).
csv_validation_data = "gold standard,generated\nCQ1? CQ2?,Real competency question: CQ1? CQ2? CQ3?\n"
validation_file_bytes = csv_validation_data.encode("utf-8")

response = client.post(
    "/validate",
    data={
        "validation_mode": "all",
        "output_folder": "test_heatmaps",
        "api_key": API_KEY,
        "model": "gpt-4"
    },
    files={"file": ("real_validation.csv", validation_file_bytes, "text/csv")}
)
print("Validation response:")
print(json.dumps(response.json(), indent=2))
