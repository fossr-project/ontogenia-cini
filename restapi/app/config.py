import os
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Global configuration settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "insert_key")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "benchmarkdataset.csv")
EXTERNAL_CQ_GENERATION_URL = os.getenv("EXTERNAL_CQ_GENERATION_URL", "http://127.0.0.1:8001/newapi") #e.g., your personal url
OUTPUTS_DIR = os.getenv(
    "OUTPUTS_DIR",
    os.path.join(ROOT_DIR, "restapi", "outputs"),
)
CQ_OUTPUTS_DIR = os.getenv(
    "CQ_OUTPUTS_DIR",
    os.path.join(OUTPUTS_DIR, "cq_validation"),
)
HEATMAP_OUTPUT_FOLDER = os.getenv(
    "HEATMAP_OUTPUT_FOLDER",
    os.path.join(CQ_OUTPUTS_DIR, "heatmaps"),
)
RESULTS_DIR = os.getenv(
    "RESULTS_DIR",
    os.path.join(CQ_OUTPUTS_DIR, "results"),
)
ONTOLOGY_DATASET_DIR = os.getenv(
    "ONTOLOGY_DATASET_DIR",
    os.path.join(ROOT_DIR, "datasets", "ontology_generation", "normalized"),
)
ONTOLOGY_RUNS_DIR = os.getenv(
    "ONTOLOGY_RUNS_DIR",
    os.path.join(OUTPUTS_DIR, "ontology_benchmark", "runs"),
)
EXTERNAL_ONTOLOGY_SERVICE_URL = os.getenv(
    "EXTERNAL_ONTOLOGY_SERVICE_URL",
    "http://127.0.0.1:8020/generate_ontology",
)
ONTOLOGY_EXTERNAL_TIMEOUT = float(os.getenv("ONTOLOGY_EXTERNAL_TIMEOUT", "300"))
OOPS_API_URL = os.getenv("OOPS_API_URL", "")
OOPS_API_MODE = os.getenv("OOPS_API_MODE", "text")  # text|file|url|xml
OOPS_API_TIMEOUT = float(os.getenv("OOPS_API_TIMEOUT", "60"))
ONTOLOGY_LLM_EVAL_PROMPT_PATH = os.getenv(
    "ONTOLOGY_LLM_EVAL_PROMPT_PATH",
    os.path.join(ROOT_DIR, "datasets", "ontology_generation", "prompts", "oe_assist_prompt.txt"),
)
ONTOLOGY_LLM_EVAL_MODEL = os.getenv("ONTOLOGY_LLM_EVAL_MODEL", OPENAI_MODEL)
ONTOLOGY_LLM_EVAL_MAX_TOKENS = int(os.getenv("ONTOLOGY_LLM_EVAL_MAX_TOKENS", "800"))
ONTOLOGY_LLM_EVAL_MAX_CHARS = int(os.getenv("ONTOLOGY_LLM_EVAL_MAX_CHARS", "12000"))
