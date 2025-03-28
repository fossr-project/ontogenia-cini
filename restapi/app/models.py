# app/models.py
from pydantic import BaseModel
from typing import Optional, List

class CQValidationResult(BaseModel):
    Gold_Standard: str
    Generated: str
    Average_Cosine_Similarity: Optional[float] = None
    Max_Cosine_Similarity: Optional[float] = None
    Average_Jaccard_Similarity: Optional[float] = None
    Cosine_Heatmap: Optional[str] = None
    Jaccard_Heatmap: Optional[str] = None
    LLM_Analysis: Optional[str] = None
    Error: Optional[str] = None

class CQValidationResponse(BaseModel):
    message: str
    results_saved_to: Optional[str] = None
    validation_results: List[CQValidationResult]

class CQGenerationResponse(BaseModel):
    csv_output: str  # CSV content as string
