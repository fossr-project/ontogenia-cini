# REST API 

This folder hosts small services to support dataset‑driven knowledge engineering with LLMs. There is an `app` subfolder for a bundled UI or deployment artifacts, a `tutorials` subfolder with hands‑on guides, and a `tests` subfolder for quick sanity checks. The code exposes two HTTP services: a Competency Question generator and a Knowledge Graph mapping generator. Each service runs independently.

## Prerequisites

Create and activate a virtual environment and install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install fastapi uvicorn flask pandas requests openpyxl openai python-dotenv
```

Set at least one provider key and optionally the model names.

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini

export TOGETHER_API_KEY=...
export TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
```

## Service: Competency Question generator

File name: `cq_generator_app.py`

Purpose: given a CSV with columns such as Scenario, Dataset, and Description, it generates a competency question per row.

Start the service:

```bash
python cq_generator_app.py
```

Default bind is `http://127.0.0.1:8001`

Endpoint is `POST /newapi/` with `multipart/form-data`. Expected fields are `file` as a CSV upload, `llm_provider` set to `openai` or `together` or `claude` with default `openai`, and optional sampling parameters. The response is a CSV with the gold standard and the generated question.

Example request:

```bash
curl -X POST "http://127.0.0.1:8001/newapi/"   -F "file=@./examples/cq_input.csv"   -F "llm_provider=openai"   --output generated_cq.csv
```

## Service: KG mapping generator

File name: `kg-generator.py`

Purpose: given a dataset and an ontology URI, it prompts an LLM to produce two artifacts aligned with the same modeling choices. One artifact is an RML mapping in Turtle. The other is a SPARQL Anything script. The dataset may be provided as an upload or by URL. Supported formats include CSV, TSV, JSON, and Excel. Parquet is not supported.

Start the service by running the file directly:

```bash
python kg-generator.py
```

The main section starts Uvicorn on `http://127.0.0.1:8010`

If you prefer the `module:app` form, rename the file to `kg_generator_app.py` and run:

```bash
uvicorn kg_generator_app:app --host 127.0.0.1 --port 8010
```

Endpoint is `POST /kg/generate` with `multipart/form-data`. Required field is `ontology_uri`. Optional fields are `provider`, `temperature`, `max_tokens`, and `dataset_url`. You may also upload a dataset via `file`. When both URL and file are given, the file is used. The service loads the dataset and samples the first five rows, fetches the ontology text, builds a prompt, sends it to the selected provider, and expects two fenced code blocks: one beginning with `RML_TTL` and one beginning with `SPARQL_ANYTHING`. The response is a zip archive containing the RML mapping, the SPARQL Anything script, and the prompt context used.

Example with a dataset URL using OpenAI:

```bash
curl -X POST "http://127.0.0.1:8010/kg/generate"   -F "ontology_uri=https://xmlns.com/foaf/spec/index.rdf"   -F "dataset_url=https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"   -F "provider=openai"   --output kg_artifacts.zip
```

Example with a file upload using Claude:

```bash
curl -X POST "http://127.0.0.1:8010/kg/generate"   -F "ontology_uri=https://w3id.org/people/foaf"   -F "file=@./examples/sample.xlsx"   -F "provider=claude"   --output kg_artifacts.zip
```

Notes: there is no server‑side validation of the LLM outputs. If you need to validate Turtle or SPARQL syntax, add a parser such as `rdflib` in your own workflow.

## Folder layout

`app` holds the bench4ke app.  
`tutorials` holds step‑by‑step walkthroughs with sample requests and expected outputs.  
`tests` holds minimal examples and small harnesses to verify endpoint reachability and environment configuration.

## Bench4KE Validator UI

File name: `bench4ke-validate-ui.py`

Purpose: a small Flask UI that calls an external Validator API to validate the CQ service and optionally display heatmaps.

Configure and run:

```bash
export CQ_API_URL="http://127.0.0.1:8000"
export CQ_API_TIMEOUT=3600
export FLASK_SECRET_KEY="changeme"
python bench4ke-validate-ui.py
```

Open `http://127.0.0.1:5000` in a browser, paste the CQ Generator URL such as `http://127.0.0.1:8001/newapi/`, and start validation.


