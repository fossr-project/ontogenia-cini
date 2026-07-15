# app/models.py
from pydantic import BaseModel, Field
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


# --- Consistency evaluation config (Pydantic v1) ---
class TestCaseConfig(BaseModel):
    enabled: bool = True
    models: List[str] = []  # generation model names; meaning depends on the test axis

class ConsistencyConfig(BaseModel):
    n_runs: int = Field(5, ge=3)                    # minimum 3 enforced
    external_service_url: str                       # required - the CQ generator (agnostic seam)
    generator_llm_provider: str = "openai"          # provider for the external generation service
    generator_temperature: Optional[float] = None   # forwarded to the generator; None -> its default
    validation_mode: str = "all"
    # validation / judge LLMs - held fixed across all runs to avoid confounding
    validation_llm_provider: str = "openai"   # provider for the validation/judge LLMs
    validation_model: str = "gpt-4"
    evaluator_llm: str = "gpt-4"
    tool_llm: Optional[str] = None
    # dataset subsetting
    gold_standard_path: Optional[str] = None    # None -> DEFAULT_DATASET
    dataset_projects: Optional[List[str]] = None
    dataset_max_rows: Optional[int] = None
    output_dir: str = "consistency_results"
    save_results: bool = False
    poll_interval: int = 10
    max_generation_workers: int = 15
    max_validation_workers: int = 1
    original_scores: Optional[dict] = None
    pass_threshold: float = -0.05
    recompute_baseline: bool = True
    repeatability: TestCaseConfig
    update_impact: TestCaseConfig
    replacement: TestCaseConfig

class ConsistencyJobResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str

class ConsistencyStatusResponse(BaseModel):
    job_id: str
    status: str     # "running" | "complete" | "failed"
    progress: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
