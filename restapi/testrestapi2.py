import json
import pandas as pd
from io import StringIO
from fastapi.testclient import TestClient
from cqver_rest_api import app  # Adjust this import if your app is in a different module

client = TestClient(app)
API_KEY = ""

# ----- Read and Filter CSV -----

csv_file_path = "benchmarkdataset.csv"
with open(csv_file_path, "rb") as f:
    file_bytes = f.read()

df = pd.read_csv(StringIO(file_bytes.decode("utf-8")))

# Keep only rows with a non-empty "Scenario" and empty "Description" and "Dataset"
filtered_df = df[
    (df["Scenario"].notna()) & (df["Scenario"].str.strip() != "") &
    ((df["Description"].isna()) | (df["Description"].str.strip() == "")) &
    ((df["Dataset"].isna()) | (df["Dataset"].str.strip() == ""))
    ]

# Get distinct scenarios
distinct_scenarios = filtered_df["Scenario"].unique()

for scenario in distinct_scenarios:
    print("\n===================================")
    print(f"Processing Scenario: {scenario}")

    # Filter the DataFrame for the current scenario
    scenario_df = filtered_df[filtered_df["Scenario"] == scenario]

    # Convert the scenario-specific DataFrame to CSV bytes
    scenario_csv = scenario_df.to_csv(index=False)
    scenario_csv_bytes = scenario_csv.encode("utf-8")

    # ----- Test Generation Endpoint (scenario mode) for this scenario -----

    response_gen = client.post(
        "/generate",
        data={
            "generation_mode": "scenario",
            "api_key": API_KEY,
            "model": "gpt-4"
        },
        files={"file": (csv_file_path, scenario_csv_bytes, "text/csv")}
    )

    gen_json = response_gen.json()
    print("Generation response:")
    print(json.dumps(gen_json, indent=2))

    generated_cqs = gen_json.get("generated_competency_questions", "")

    # ----- Prepare CSV for CQ Validation for this scenario -----
    # Use the "Competency Question" column as the gold standard
    # If there are multiple entries, join them with a separator
    gold_standard = "; ".join(scenario_df["Competency Question"].dropna().unique())

    # Create a new DataFrame for validation with one row for this scenario
    val_df = pd.DataFrame({
        "gold standard": [gold_standard],
        "generated": [generated_cqs]
    })

    scenario_val_csv = val_df.to_csv(index=False)
    scenario_val_csv_bytes = scenario_val_csv.encode("utf-8")

    # ----- Test Validation Endpoint for this scenario -----

    response_val = client.post(
        "/validate",
        data={
            "validation_mode": "all",
            "output_folder": "test_heatmaps",
            "api_key": API_KEY,
            "model": "gpt-4"
        },
        files={"file": (csv_file_path, scenario_val_csv_bytes, "text/csv")}
    )

    print("Validation response:")
    print(json.dumps(response_val.json(), indent=2))
