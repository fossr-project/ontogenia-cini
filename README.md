<p align="center">
  <img src="./benchmarklogo.png" alt="Bench4KE Logo" width="500"/>
</p>

<h1 align="center">Bench4KE</h1>
<h3 align="center"><i>A Benchmarking System for Evaluating Knowledge Engineering Automation Tasks</i></h3>

<p align="center">
  <a href="https://github.com/fossr-project/ontogenia-cini"><img src="https://img.shields.io/badge/website-Bench4KE-blue?style=plastic" alt="Website"></a>
  <a href="https://github.com/fossr-project/ontogenia-cini/blob/main/restapi/tutorial/Bench4KE%20Tutorial.pdf"><img src="https://img.shields.io/badge/doc-API_Tutorial-dodgerblue?style=plastic" alt="API"></a>
  <a href="https://docs.google.com/forms/d/e/1FAIpQLSfpYHGzA2r0wKCq0xEVIkPBKKol6umiKn1URAc17f709DKMKg/viewform?usp=header"><img src="https://img.shields.io/badge/link-Evaluation_Questionnaire-deepskyblue?style=plastic" alt="Questionnaire"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-APACHE-00BCD4?style=plastic" alt="License"></a>
</p>

**Bench4KE** is a benchmarking framework designed to evaluate KE automation with Large Language Models. It supports both Competency Question (CQ) generation evaluation and ontology generation benchmarking.

CQs are natural language questions used by ontology engineers to define and validate the functional requirements of an ontology. With the increasing use of LLMs to automate tasks in Knowledge Engineering, the automatic generation of CQs is gaining attention. However, current evaluation approaches lack standardization and reproducibility.

**Bench4KE** addresses this gap by providing:

## Key Features

- A gold standard dataset derived from real-world ontology engineering projecs  
- Multiple evaluation metrics:
  - Cosine Similarity
  - BERTScore-F1
  - Jaccard Similarity
  - BLEU
  - ROUGE-L
  - Hit Rate
  - LLM-based semantic analysis (via OpenAI models)
- Visual heatmaps for comparing generated and manually crafted CQs
- Ontology generation benchmarking:
  - OntoMetrics-style structural metrics
  - OOPS! pitfall detection
  - LLM-based evaluation (OE-Assist prompt with yes/no + SPARQL)
- Modular and extensible architecture to support the upload of a custom dataset, additional KE tasks and other evaluation metrics in the future

## Directory Contents

| File / Folder             | Description |
|--------------------------|-------------|
| `app/`                   | Contains the FastAPI application modules and related components. |
| `benchmarkdataset.csv`   | This is the gold standard dataset of manually crafted CQs used for evaluation. |
| `tests/`                 | Directory for test cases and testing utilities. |
| `tutorial/`              | Tutorial materials to use the API. |
| `cq_generator_app.py`    | Example of a CQ generation application that is compatible with the API. |
| `bench4ke-validate-ui.py`| Web interface built on the API. |

## Configuration (.env)

Create a `.env` file in the repo root by copying the example:

```bash
cp .env.example .env
```

Then edit the values you need (see comments inside `.env.example`).

Essentials:
- `OPENAI_API_KEY` (required for any real run)
- `OOPS_API_URL` is required only if you want OOPS enabled; leave empty to skip it. For local Docker use `http://localhost:8080/OOPS/rest`.

Optional with defaults:
- `OPENAI_MODEL` defaults to `gpt-4o-mini`
- `EXTERNAL_CQ_GENERATION_URL` defaults to `http://127.0.0.1:8001/newapi`
- `EXTERNAL_ONTOLOGY_SERVICE_URL` defaults to `http://127.0.0.1:8020/generate_ontology`
- `ONTOLOGY_SYSTEM` defaults to `ontogenia`

Activate the `.env` before running services:

```bash
source .env
```

Most services also load `.env` automatically via `python-dotenv`, but exporting it explicitly keeps CLI runs consistent.

### OOPS (local Docker + WordNet)
OOPS requires WordNet to be mounted in the container at:
`/usr/local/tomcat/WordNet/WordNet-3.0/dict/index.sense`.

Ensure your repo has:
```
WordNet/WordNet-3.0/dict/index.sense
```

Run OOPS locally:
```bash
docker run --rm -v "$PWD/WordNet:/usr/local/tomcat/WordNet" -p 8080:8080 mpovedavillalon/oops:v1
```

Set the endpoint for the benchmark:
```
OOPS_API_URL=http://localhost:8080/OOPS/rest
```

## Usage

To evaluate a CQ Generation tool or an ontology generation system using **Bench4KE**, follow the steps below:

### 1. Setup

Ensure you have Python 3.8 or higher installed. 

### 2. Install Dependencies 

Download the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Services

Start the Bench4KE API:
```bash
cd restapi
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Start the CQ generator service (for CQ validation):
```bash
cd restapi
python cq_generator_app.py
```

Start the ontology adapter (for ontology benchmarking):
```bash
cd restapi
python ontology_adapter.py
```

### 4. CQ Validation
Run CQ validation via API:

```bash
curl -X POST http://127.0.0.1:8000/validate/ \
  -F use_default_dataset=true \
  -F external_service_url=http://127.0.0.1:8001/newapi \
  -F validation_mode=all \
  -F save_results=true
```

Results are saved under:
```
restapi/outputs/cq_validation/
```

### 5. Ontology Benchmark
Run ontology benchmarking via API:

```bash
curl -X POST http://127.0.0.1:8000/ontology/run \
  -H "Content-Type: application/json" \
  -d '{
    "use_default_dataset": true,
    "evaluation_mode": "all",
    "external_service_url": "http://127.0.0.1:8020/generate_ontology",
    "save_results": true
  }'
```

Results are saved under:
```
restapi/outputs/ontology_benchmark/
```

### 6. Web UI
Launch the UI:
```bash
cd restapi
python bench4ke-validate-ui.py
```
Open `http://127.0.0.1:5000` and use the CQ Validation or Ontology Benchmark tabs.

## Citation
```
@misc{bench4ke_2025,
  title        = {{Bench4KE}: A Benchmarking System for Evaluating LLM-based Competency Question Generation},
  howpublished = {\url{https://github.com/fossr-project/ontogenia-cini}},
  note         = {Commit accessed 29~Apr~2025},
  year         = {2025}
}
```

## License
Licensed under the [Apache License](./LICENSE).
