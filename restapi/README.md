# CQ Verification: usage

This project provides an API to verify competency questions (CQs) by comparing a gold standard with generated questions. The verification process computes similarity metrics (cosine and Jaccard) and generates heatmaps, with GPT-4. 

## Project Structure

- **cq_verification.py**  
  Contains functions for:
  - Extracting and processing competency questions.
  - Calculating cosine and Jaccard similarity metrics.
  - Generating heatmaps (returned as base64-encoded PNGs and optionally saved to disk).

- **updated-restapi.py**  
  Implements the FastAPI REST API that:
  - Accepts custom CSV files or uses a default dataset.
  - Calls an external CQ generation service
  - Validates competency questions using the functions in `cq_verification.py`.
  - Saves validation results (including heatmaps) in a results folder.

- **test-restapi3.py**  
  A test script that demonstrates how to:
  - Call the API using the default dataset.
  - call the API using a custom dataset

- **cq-gen-api-to-test.py**  
  A fake external CQ generation service built with FastAPI. It:
  - Accepts a CSV file with a `"gold standard"` or `"Competency Question"` column.
  - Generates competency questions via GPT-4.
  - Returns a CSV with an added `"generated"` column.

## External CQ Generation Service Requirements

For the CQ verification process to work correctly, the external CQ generation service must adhere to these guidelines:

- **Input Requirements**:
  - Accept a CSV file (via file upload) that contains at least one of the following columns:
    - `"gold standard"` **OR**
    - `"Competency Question"`
  - The CSV should have one row per competency question entry.

- **Output Requirements**:
  - Return a CSV file that includes all original columns plus a new column:
    - `"generated"` — containing the generated competency questions.
  - The generated questions must correspond in order to the input gold standard questions.

## CSV Dataset Format

When not using the default dataset (`use_default_dataset=False`), the uploaded CSV file must include:
- A column named **either** `"gold standard"` **or** `"Competency Question"`.
- Optionally, any additional columns can be present.
- Each row represents a single competency question entry.


