import os
import io
import re
import csv
import json
import uuid
import logging
import zipfile
import requests
import pandas as pd
from typing import Optional, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import StreamingResponse
from io import StringIO, BytesIO
import openai

# -----------------------------------------------------------------------------
# Config & logging
# -----------------------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY", "")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("kg-generator")

app = FastAPI(
    title="KG Generator Service",
    description="Generate RML mappings and SPARQL Anything scripts from a dataset and an ontology URI via LLM prompting.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Helpers: dataset loading and preview
# -----------------------------------------------------------------------------
SUPPORTED_EXTS = {".csv", ".tsv", ".txt", ".json", ".xlsx", ".xls"}
PARQUET_EXTS = {".parquet", ".pq"}

def _ext(path: str) -> str:
    return os.path.splitext(path.split("?")[0].split("#")[0])[1].lower()

def fetch_bytes(path_or_url: str) -> bytes:
    # GitHub tree URL to raw conversion (same idea as your other service)
    if path_or_url.startswith("https://github.com/") and "/tree/" in path_or_url:
        parts = path_or_url.split("/tree/")
        repo_url, rest = parts[0], parts[1]
        raw_base = repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        path_or_url = f"{raw_base}/{rest}"
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return r.content
    if not os.path.isfile(path_or_url):
        raise FileNotFoundError(f"Local file not found: {path_or_url}")
    with open(path_or_url, "rb") as f:
        return f.read()

def load_dataframe_from_any(data: bytes, filename_hint: str) -> pd.DataFrame:
    ext = _ext(filename_hint)
    if ext in PARQUET_EXTS:
        raise HTTPException(status_code=400, detail="Parquet is not supported by design.")
    if ext not in SUPPORTED_EXTS:
        # Try to sniff simple CSV/TSV by content as a fallback
        try:
            s = data.decode("utf-8", errors="ignore")
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(s.splitlines()[0])
            df = pd.read_csv(StringIO(s), dialect=dialect)
            return df
        except Exception:
            raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext or '[none]'}")

    try:
        if ext in {".csv", ".txt"}:
            return pd.read_csv(io.BytesIO(data))
        if ext == ".tsv":
            return pd.read_csv(io.BytesIO(data), sep="\t")
        if ext == ".json":
            obj = json.loads(data.decode("utf-8", errors="ignore"))
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                return pd.DataFrame(obj["data"])
            return pd.json_normalize(obj)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(io.BytesIO(data))
    except Exception as e:
        logger.exception("Failed to parse dataset")
        raise HTTPException(status_code=400, detail=f"Error parsing dataset: {e}")
    raise HTTPException(status_code=400, detail=f"Unsupported input format: {ext}")

def build_schema_summary(df: pd.DataFrame, sample_rows: int = 5) -> Tuple[str, str]:
    sample = df.head(sample_rows).copy()
    # Create a type hint summary
    types = {}
    for col in sample.columns:
        dtype = str(sample[col].dtype)
        # Simplify dtypes for prompt readability
        if "int" in dtype:
            t = "integer"
        elif "float" in dtype:
            t = "float"
        elif "bool" in dtype:
            t = "boolean"
        elif "datetime" in dtype or "date" in dtype:
            t = "datetime"
        else:
            t = "string"
        types[col] = t
    schema_lines = [f"- {c}: {t}" for c, t in types.items()]
    schema = "Columns and inferred types:\n" + "\n".join(schema_lines)

    # CSV sample for prompt
    csv_sample = sample.to_csv(index=False)
    return schema, csv_sample

# -----------------------------------------------------------------------------
# Helpers: ontology fetching
# -----------------------------------------------------------------------------
def fetch_ontology_text(uri: str, max_chars: int = 60000) -> str:
    r = requests.get(uri, timeout=60)
    r.raise_for_status()
    text = r.text
    if len(text) > max_chars:
        return text[:max_chars] + "\n# [truncated for prompt]\n"
    return text

# -----------------------------------------------------------------------------
# Helpers: LLM providers
# -----------------------------------------------------------------------------
def call_llm(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2000
) -> str:
    provider = provider.lower().strip()
    logger.debug(f"LLM provider selected: {provider}")

    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("OpenAI error")
            raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    if provider == "together":
        try:
            api_key = os.getenv("TOGETHER_API_KEY", "")
            model = os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            r = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.exception("Together.ai error")
            raise HTTPException(status_code=500, detail=f"Together.ai error: {e}")

    if provider == "claude":
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"].strip()
        except Exception as e:
            logger.exception("Claude error")
            raise HTTPException(status_code=500, detail=f"Claude error: {e}")

    raise HTTPException(status_code=400, detail=f"Unknown LLM provider: {provider}")

# -----------------------------------------------------------------------------
# Prompt template
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a careful ontology and data integration assistant.
Given an ontology and a tabular dataset sample, you produce two artifacts:
1) An RML mapping document (Turtle) that maps the dataset to the ontology.
2) A SPARQL Anything script that can materialize triples conforming to the same ontology.

Guidelines:
- Use classes and properties from the provided ontology when possible.
- Assume the dataset is tabular; use RML logical sources accordingly.
- For RML, emit prefixes and use rr:/rml:/ql:/rmlt:/fnml:/fno: as appropriate.
- For SPARQL Anything, target a SELECT or CONSTRUCT workflow to generate triples; prefer CONSTRUCT when creating RDF.
- Keep URIs stable and deterministic, building subject IRIs from stable keys or row identifiers.
- Do not add explanatory prose in the code blocks.
- Ensure both artifacts reflect the same modeling decisions.
"""

USER_PROMPT_TEMPLATE = """Ontology URI:
{ontology_uri}

Ontology excerpt or content:
{ontology_text}
Dataset schema summary:
{schema_summary}

First 5 rows as CSV:
{csv_sample}

Task:
1) Generate an RML mapping document in Turtle that maps the above dataset to the ontology.
2) Generate a SPARQL Anything script that materializes the same triples for the same ontology.

Output format:
Return exactly two fenced code blocks, in ANY order:
- The RML mapping code block MUST start with a line that contains exactly: RML_TTL
- The SPARQL Anything script code block MUST start with a line that contains exactly: SPARQL_ANYTHING

Example of formatting (not content):
RML_TTL
@prefix rr: <...> .
...
SPARQL_ANYTHING
PREFIX fx: <...>
CONSTRUCT { ... }
WHERE { ... }

Do not include additional commentary outside the two code blocks.
"""

# -----------------------------------------------------------------------------
# Extraction: parse the two code blocks out of the LLM answer
# -----------------------------------------------------------------------------
RML_PATTERN = re.compile(r"```(?:[^\n]*\n)?RML_TTL\s+(.*?)```", re.DOTALL | re.IGNORECASE)
SPARQLX_PATTERN = re.compile(r"```(?:[^\n]*\n)?SPARQL_ANYTHING\s+(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_artifacts(text: str) -> Tuple[str, str]:
    rml_match = RML_PATTERN.search(text)
    sparql_match = SPARQLX_PATTERN.search(text)
    if not rml_match or not sparql_match:
        raise HTTPException(status_code=502, detail="LLM did not return both required code blocks.")
    rml = rml_match.group(1).strip()
    sparql = sparql_match.group(1).strip()
    if not rml:
        raise HTTPException(status_code=502, detail="Empty RML block from LLM.")
    if not sparql:
        raise HTTPException(status_code=502, detail="Empty SPARQL Anything block from LLM.")
    return rml, sparql

# -----------------------------------------------------------------------------
# API: POST /kg/generate
# -----------------------------------------------------------------------------
@app.post("/kg/generate")
def generate_kg(
    ontology_uri: str = Form(..., description="HTTP(S) URI of the ontology (TTL/RDF/XML/RDFS/OWL)."),
    provider: str = Form("openai", description="LLM provider: openai | together | claude"),
    temperature: float = Form(0.2),
    max_tokens: int = Form(2000),
    dataset_url: Optional[str] = Form(None, description="Optional URL to the dataset file."),
    file: Optional[UploadFile] = File(None, description="Optional uploaded dataset file. If provided, overrides dataset_url."),
):
    if not dataset_url and not file:
        raise HTTPException(status_code=400, detail="Provide either dataset_url or file.")

    # Load dataset
    try:
        if file is not None:
            raw = file.file.read()
            fname = file.filename or "uploaded.csv"
        else:
            raw = fetch_bytes(dataset_url)
            fname = dataset_url
        df = load_dataframe_from_any(raw, fname)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Dataset load failed")
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    # Prepare prompt materials
    schema_summary, csv_sample = build_schema_summary(df, sample_rows=5)

    try:
        ontology_text = fetch_ontology_text(ontology_uri)
    except Exception as e:
        logger.exception("Ontology fetch failed")
        raise HTTPException(status_code=400, detail=f"Failed to fetch ontology: {e}")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        ontology_uri=ontology_uri,
        ontology_text=ontology_text,
        schema_summary=schema_summary,
        csv_sample=csv_sample
    )

    # Call the LLM
    answer = call_llm(
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Extract artifacts
    rml_ttl, sparql_any = extract_artifacts(answer)

    # Package as zip
    zbuf = BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        base = f"kg_artifacts_{uuid.uuid4().hex[:8]}"
        zf.writestr(f"{base}.rml.ttl", rml_ttl)
        zf.writestr(f"{base}.sparql", sparql_any)
        # Also include a copy of the prompt context for reproducibility
        zf.writestr(f"{base}.prompt.txt", user_prompt)

    zbuf.seek(0)
    return StreamingResponse(
        zbuf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="kg_artifacts.zip"'}
    )

# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "KG Generator Service",
        "status": "ok",
        "endpoints": {
            "POST /kg/generate": {
                "form fields": [
                    "ontology_uri (str, required)",
                    "provider (openai|together|claude, default=openai)",
                    "temperature (float, default=0.2)",
                    "max_tokens (int, default=2000)",
                    "dataset_url (str, optional)",
                    "file (UploadFile, optional; overrides dataset_url)"
                ],
                "returns": "application/zip containing .rml.ttl and .sparql files"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("kg_generator_app:app", host="127.0.0.1", port=8010, reload=False)
