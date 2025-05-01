import os

# Global configuration settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "insert_key")
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "benchmarkdataset.csv")
EXTERNAL_CQ_GENERATION_URL = os.getenv("EXTERNAL_CQ_GENERATION_URL", "http://127.0.0.1:8001/newapi") #e.g., your personal url
HEATMAP_OUTPUT_FOLDER = os.getenv("HEATMAP_OUTPUT_FOLDER", "heatmaps")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
