from fastapi import FastAPI, File, UploadFile, HTTPException, Response
import pandas as pd
import openai
import logging
from io import StringIO
import os
import requests
import json

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "yourkey")

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(
    title="CQ Generator Service",
    description="API to generate competency questions from a CSV file containing scenarios, descriptions, and/or datasets.",
    version="1.0.0"
)

# CQ generation patterns and stylistic instructions
patterns = [
    {"pattern": "Which [class expression 1][object property expression][class expression 2]?", "example": "Which pizzas contain pork?"},
    {"pattern": "How much does [class expression][datatype property]?", "example": "How much does Margherita Pizza weigh?"},
    {"pattern": "What type of [class expression] is [individual]?", "example": "What type of software (API, Desktop application etc.) is it?"},
    {"pattern": "Is the [class expression 1][class expression 2]?", "example": "Is the software open source development?"},
    {"pattern": "What [class expression] has the [numeric modifier][datatype property]?", "example": "What pizza has the lowest price?"},
    {"pattern": "Which are [class expressions]?", "example": "Which are gluten-free bases?"}
]

instructions = [
    {
        "instruction": "Do not make explicit references to the dataset or its variables in the generated competency questions.",
        "example": {
            "incorrect": "How many cases of COVID-19 were registered in Italy in 2021?",
            "correct": "How many cases of the pathology were registered in the country in a given period?"
        }
    },
    {
        "instruction": "Keep the questions simple. Each competency question should not contain another simpler competency question within it.",
        "example": {
            "incorrect": "What is the capital of Italy and how many inhabitants does it have?",
            "correct": ["What is the capital of the country?", "How many inhabitants does the city have?"]
        }
    },
    {
        "instruction": "Do not include real entities; instead, abstract them into more generic concepts.",
        "example": {
            "incorrect": "When was Leonardo da Vinci born?",
            "correct": "When was the artist born?"
        }
    }
]

clustering_instruction = "Once the competency questions have been generated, they should be clustered into thematic areas. Each cluster represents an ontological module."


def get_dataset_bytes(path: str) -> bytes:
    """Retrieve raw CSV bytes from local file, HTTP(S) URL, or GitHub tree URL."""
    if path.startswith("https://github.com/") and "/tree/" in path:
        parts = path.split("/tree/")
        repo_url, rest = parts[0], parts[1]
        raw_base = repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        path = f"{raw_base}/{rest}"
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path)
        resp.raise_for_status()
        return resp.content
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Local dataset file not found: {path}")
    with open(path, "rb") as f:
        return f.read()

@app.post("/newapi/")
async def generate_cqs_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        logger.error(f"CSV read error: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading input CSV: {e}")

    gold_list = []
    generated_list = []

    for idx, row in df.iterrows():
        desc = row.get("Description", None)
        scen = row.get("Scenario", None)
        dpath = row.get("Dataset", None)
        gold = row.get("Competency Question", "")

        has_desc = pd.notna(desc) and str(desc).strip() != ""
        has_scen = pd.notna(scen) and str(scen).strip() != ""
        has_data = pd.notna(dpath) and str(dpath).strip() != ""

        if has_data and not has_scen and not has_desc:
            mode = "dataset"
        elif has_data and has_desc and not has_scen:
            mode = "dataset+description"
        elif has_scen and not has_data and not has_desc:
            mode = "scenario"
        else:
            logger.warning(f"Row {idx} skipped; unsupported input combination (desc={has_desc}, scen={has_scen}, data={has_data})")
            continue

        sample_csv = ""
        if mode in ("dataset", "dataset+description"):
            try:
                data_bytes = get_dataset_bytes(str(dpath))
                if str(dpath).startswith("http"):
                    data_str = data_bytes.decode("utf-8")
                    sample_df = pd.read_csv(StringIO(data_str)).head(5)
                else:
                    sample_df = pd.read_csv(str(dpath)).head(5)
                sample_csv = sample_df.to_csv(index=False)
            except Exception as e:
                logger.warning(f"Could not load dataset sample for row {idx}: {e}; proceeding without sample.")
                sample_csv = ""

        # Unified advanced prompt for all modes
        if mode == "scenario":
            user_input = f"Scenario description: {scen}"
        elif mode == "dataset":
            user_input = f"Dataset description: No description provided\n\nDataset sample:\n{sample_csv}"
        elif mode == "dataset+description":
            user_input = f"Dataset description: {desc}\n\nDataset sample:\n{sample_csv}"

        messages = [
            {"role": "system", "content": "You are an ontology engineer. Generate a list of competency questions based on the provided input, following these patterns and instructions."},
            {"role": "user", "content": user_input},
            {"role": "system", "content": f"Use the following competency question patterns:\n{json.dumps(patterns, indent=2)}"},
            {"role": "system", "content": f"Follow these instructions when generating the competency questions:\n{json.dumps(instructions, indent=2)}"},
            {"role": "system", "content": f"After generating the questions, cluster them into thematic areas according to these guidelines:\n{clustering_instruction}"}
        ]

        try:
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0,
                max_tokens=500
            )
            gen = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error at row {idx}: {e}")
            gen = f"Error generating CQ: {e}"

        gold_list.append(gold)
        generated_list.append(gen)

    result_df = pd.DataFrame({
        "gold standard": gold_list,
        "generated": generated_list
    })
    return Response(content=result_df.to_csv(index=False), media_type="text/csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cq_generator_app:app", host="127.0.0.1", port=8001, reload=True)
