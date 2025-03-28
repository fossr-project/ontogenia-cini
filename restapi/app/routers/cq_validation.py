from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
from io import StringIO
import os, time

from app.services.cq_validator import CQValidator
from app.utils.external_call import call_external_cq_generation_service
from app.config import DEFAULT_DATASET, RESULTS_DIR

router = APIRouter()

@router.post("/")
async def validate_competency_questions(
    file: UploadFile = File(None),
    validation_mode: str = Form("all"),
    output_folder: str = Form("heatmaps"),
    use_default_dataset: bool = Form(False),
    external_service_url: str = Form(...),
    api_key: str = Form(None),
    model: str = Form("gpt-4"),
    save_results: bool = Form(True)
):
    """
    Endpoint to validate competency questions.
    """
    if file is not None:
        try:
            contents = await file.read()
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")
    elif use_default_dataset:
        try:
            df = pd.read_csv(DEFAULT_DATASET)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading default dataset: {e}")
    else:
        raise HTTPException(status_code=400, detail="Either upload a CSV file or set use_default_dataset=True.")

    if "gold standard" not in df.columns and "Competency Question" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'gold standard' or 'Competency Question' column.")

    # If 'generated' column is missing, call external CQ generation service
    if "generated" not in df.columns:
        try:
            df = call_external_cq_generation_service(df, external_service_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling external CQ generation service: {e}")
    df = df.head(5) #CHANGE IF YOU WANT TO USE THE FULL DATASET
    gold_col = "gold standard" if "gold standard" in df.columns else "Competency Question"

    validator = CQValidator(output_folder=output_folder, model=model, validation_mode=validation_mode)
    results = []
    for idx, row in df.iterrows():
        try:
            result = validator.validate(row[gold_col], row["generated"])
            results.append({
                "Gold Standard": row[gold_col],
                "Generated": row["generated"],
                **result
            })
        except Exception as e:
            results.append({
                "Gold Standard": row[gold_col],
                "Generated": row["generated"],
                "Error": str(e)
            })

    results_file = ""
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = int(time.time())
        results_file = os.path.join(RESULTS_DIR, f"validation_results_{timestamp}.csv")
        import csv
        keys = set()
        for r in results:
            keys.update(r.keys())
        keys = list(keys)

        with open(results_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    import math

    def clean_nans(obj):
        if isinstance(obj, list):
            return [clean_nans(o) for o in obj]
        elif isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    return JSONResponse(content={
        "message": "Processing complete",
        "results_saved_to": results_file if save_results else "Not saved",
        "validation_results": clean_nans(results)
    })

