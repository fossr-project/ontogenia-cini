from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import re
import logging
from io import StringIO
import requests
from typing import Optional
from cq_verification import validate_cq

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def remove_html_tags(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', text)

app = FastAPI(
    title="CQ Verification API",
    description=(
        "API for verifying competency questions. The endpoint expects a CSV file that contains at least the "
        "gold standard competency questions (in a column such as 'gold standard' or 'Competency Question'). "
        "If the CSV does not include a 'generated' column, the API calls an external CQ generation service. "
        "That external service must accept the benchmark CSV and return a CSV with a 'generated' column added. "
        "Then the API computes similarity metrics between the gold standard and generated CQs."
    ),
    version="1.0.0"
)

def call_external_cq_generation_service(df: pd.DataFrame, external_service_url: str) -> pd.DataFrame:
    """
    Call an external CQ generation service to generate competency questions.

    This function sends the benchmark CSV (which must have a gold standard column) to an external service,
    which returns a CSV containing the generated competency questions (in a 'generated' column). The function
    then merges that result into the original DataFrame.
    """
    csv_data = df.to_csv(index=False)
    files = {
        "file": ("benchmarkdataset.csv", csv_data, "text/csv")
    }
    try:
        response = requests.post(external_service_url, files=files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling external CQ generation service: {e}")

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"External CQ generation service error: {response.text}")

    try:
        df_generated = pd.read_csv(StringIO(response.text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading response CSV from external service: {e}")

    if "generated" not in df_generated.columns:
        raise HTTPException(status_code=500, detail="The external service did not return a 'generated' column.")

    df["generated"] = df_generated["generated"]
    return df


import os
import time
import csv


@app.post("/validate")
async def validate_competency_questions(
        file: Optional[UploadFile] = File(None),
        validation_mode: str = Form("all"),
        output_folder: str = Form("heatmaps"),
        use_default_dataset: bool = Form(False),
        external_service_url: str = Form(...),
        api_key: Optional[str] = Form(None),
        model: str = Form("gpt-4"),
        save_results: bool = Form(True)  # New parameter to enable saving results
):
    """
    Endpoint to validate competency questions and save results (if enabled).
    """

    if file is not None:
        try:
            contents = await file.read()
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
            df = df.head(5)  # REMOVE THIS LIMIT FOR FULL PROCESSING
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")
    elif use_default_dataset:
        try:
            df = pd.read_csv("benchmarkdataset.csv")
            df = df.head(5)  # REMOVE THIS LIMIT FOR FULL PROCESSING
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading default dataset: {e}")
    else:
        raise HTTPException(status_code=400, detail="Either upload a CSV file or set use_default_dataset=True.")

    if "gold standard" not in df.columns and "Competency Question" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'gold standard' or 'Competency Question'.")

    if "generated" not in df.columns:
        try:
            df = call_external_cq_generation_service(df, external_service_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling external CQ generation service: {e}")

    gold_col = "gold standard" if "gold standard" in df.columns else "Competency Question"

    results = []
    for idx, row in df.iterrows():
        input_text = f"Gold standard: {row[gold_col]}\nGenerated: {row['generated']}"

        try:
            result_text = validate_cq(input_text, mode=validation_mode, output_folder=output_folder, model=model)
            clean_result = remove_html_tags(result_text)

            # Parse the result to extract metrics
            cosine_match = re.search(r"Average cosine: ([\d.]+), Max cosine: ([\d.]+)", result_text)
            jaccard_match = re.search(r"Average jaccard: ([\d.]+)", result_text)

            avg_cosine = float(cosine_match.group(1)) if cosine_match else None
            max_cosine = float(cosine_match.group(2)) if cosine_match else None
            avg_jaccard = float(jaccard_match.group(1)) if jaccard_match else None

            # Extract heatmap file paths
            cosine_file = re.search(r"Cosine heatmap saved to: ([^<]+)", result_text)
            jaccard_file = re.search(r"Jaccard heatmap saved to: ([^<]+)", result_text)

            cosine_path = cosine_file.group(1) if cosine_file else "N/A"
            jaccard_path = jaccard_file.group(1) if jaccard_file else "N/A"

            results.append({
                "Gold Standard": row[gold_col],
                "Generated": row["generated"],
                "Average Cosine Similarity": avg_cosine,
                "Max Cosine Similarity": max_cosine,
                "Average Jaccard Similarity": avg_jaccard,
                "Cosine Heatmap": cosine_path,
                "Jaccard Heatmap": jaccard_path,
                "LLM Analysis": clean_result
            })

        except Exception as e:
            results.append({
                "Gold Standard": row[gold_col],
                "Generated": row["generated"],
                "Error": str(e)
            })

    # Save results to a CSV file if enabled
    if save_results:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = int(time.time())
        results_file = os.path.join(results_dir, f"validation_results_{timestamp}.csv")

        with open(results_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    return JSONResponse(content={
        "message": "Processing complete",
        "results_saved_to": results_file if save_results else "Not saved",
        "validation_results": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
