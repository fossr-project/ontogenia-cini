from fastapi import FastAPI, File, UploadFile, HTTPException, Response
import pandas as pd
import openai
import logging
from io import StringIO
import os
import requests

openai.api_key = os.getenv("OPENAI_API_KEY", "yourkey")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CQ Generator Service",
    description="Standalone API to generate competency questions from an input CSV with scenario/dataset/description.",
    version="1.0.0"
)

def get_dataset_bytes(path: str) -> bytes:
    """
    Fetch raw CSV bytes from a local path, HTTP URL, or GitHub repository URL.
    If detecting a GitHub tree URL, converts it to raw.githubusercontent.com format.
    """
    # Handle GitHub tree URLs: convert to raw content URL
    if path.startswith("https://github.com/") and "/tree/" in path:
        parts = path.split("/tree/")
        repo_url, rest = parts[0], parts[1]
        raw_base = repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        path = f"{raw_base}/{rest}"
    # Fetch remote
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path)
        resp.raise_for_status()
        return resp.content
    # Local fallback
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Local dataset file not found: {path}")
    with open(path, "rb") as f:
        return f.read()

@app.post("/newapi/")
async def generate_cqs_endpoint(
    file: UploadFile = File(...)
):
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
            logger.warning(f"Row {idx} skipping; unsupported mode (desc={has_desc}, scen={has_scen}, data={has_data})")
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

        if mode == "scenario":
            prompt = f"Generate a competency question from this scenario:\n\"{scen}\""
        elif mode == "dataset+description":
            if sample_csv:
                prompt = f"Generate a competency question using the following description and dataset sample:\nDescription: {desc}\nSample:\n{sample_csv}"
            else:
                prompt = f"Generate a competency question using the following description and dataset at: {dpath}\nDescription: {desc}"
        else:  
            if sample_csv:
                prompt = f"Generate a competency question from this dataset sample:\n{sample_csv}"
            else:
                prompt = f"Generate a competency question based on the dataset located at: {dpath}"

        try:
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a competency question generation assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.5
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
