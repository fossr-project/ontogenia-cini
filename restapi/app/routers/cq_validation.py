from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
from io import StringIO
import os
import logging

from app.utils.external_call import call_external_cq_generation_service
from app.services.validation_pipeline import (
    run_validation_pipeline,
    normalize_cq_columns,
)
from app.config import DEFAULT_DATASET

logger = logging.getLogger(__name__)
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
    save_results: bool = Form(True),
    save_every: int = Form(1),
    evaluator_llm: str = Form("gpt-4"),
    tool_llm: str = Form(None),
    generator_llm_provider: str = Form("openai"),
    generator_model: str = Form(None),
    generated_csv_path: str = Form(None),
):
    """Validate competency questions against a gold standard benchmark."""
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

    # Normalize CQ column names and detect the gold column (shared helper).
    df, gold_col = normalize_cq_columns(df)
    if gold_col is None:
        raise HTTPException(status_code=400, detail="CSV must contain 'gold standard' or 'Competency Question' column.")

    if "generated" not in df.columns:
        # Resume from a previously saved generation file if provided
        if generated_csv_path and os.path.exists(generated_csv_path):
            try:
                df_gen = pd.read_csv(generated_csv_path)
                df["generated"] = df_gen["generated"].values
                logger.info("Loaded generated CQs from %s (%d rows)", generated_csv_path, len(df_gen))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error loading generated_csv_path: {e}")
        else:
            try:
                df = call_external_cq_generation_service(
                    df, external_service_url,
                    llm_provider=generator_llm_provider,
                    model=generator_model,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error calling external CQ generation service: {e}")

    content = run_validation_pipeline(
        df=df,
        gold_col=gold_col,
        output_folder=output_folder,
        model=model,
        validation_mode=validation_mode,
        save_results=save_results,
        save_every=save_every,
        evaluator_llm=evaluator_llm,
        tool_llm=tool_llm,
    )
    return JSONResponse(content=content)
