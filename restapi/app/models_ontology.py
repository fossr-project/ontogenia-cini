from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OntologyConstraints(BaseModel):
    output_format: Optional[str] = Field(default="ttl", description="ttl|rdfxml|owl|jsonld")
    iri_base: Optional[str] = None
    naming_policy: Optional[str] = None
    language: Optional[str] = None


class OntologyGenerationItem(BaseModel):
    system: Optional[str] = None
    dataset_id: Optional[str] = None
    scenario_id: Optional[str] = None
    scenario: Optional[str] = None
    competency_questions: List[str]
    user_stories: Optional[List[str]] = None
    constraints: Optional[OntologyConstraints] = None
    metadata: Optional[Dict[str, Any]] = None


class OntologyBenchmarkRequest(BaseModel):
    system: Optional[str] = Field(default=None, description="ontogenia|ontogenia-mp|domain-ontogen|neon-gpt|neon-gpt-llms4life|all")
    use_default_dataset: bool = False
    dataset_path: Optional[str] = None
    items: Optional[List[OntologyGenerationItem]] = None
    external_service_url: Optional[str] = None
    model: Optional[str] = None
    evaluation_mode: str = Field(default="all", description="all|ontometrics|oops|llm or comma-separated")
    llm_eval_model: Optional[str] = None
    domain_ontogen_mode: str = Field(
        default="per_item",
        description="How to handle Domain-OntoGen items with multiple CQs: per_item (merge) or per_cq (independent ontology per CQ, paper-style).",
    )
    max_items: int = 0
    save_results: bool = True


class OntologyRunItemResult(BaseModel):
    dataset_id: Optional[str] = None
    system: Optional[str] = None
    ontology_file: Optional[str] = None
    ontometrics_file: Optional[str] = None
    oops_file: Optional[str] = None
    llm_eval_file: Optional[str] = None
    llm_eval_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OntologyBenchmarkResponse(BaseModel):
    message: str
    run_dir: Optional[str] = None
    results_saved_to: Optional[str] = None
    results: List[OntologyRunItemResult]
