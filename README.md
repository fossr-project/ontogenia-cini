<h1 align="center">CQ-Verify</h1>
<h3 align="center"><i>A Benchmarking System for Evaluating LLM-based Competency Question Generation</i></h3>

**CQ-Verify** is a benchmarking framework designed to evaluate the quality of Competency Questions automatically generated by Large Language Models.

CQs are natural language questions used by ontology engineers to define and validate the functional requirements of an ontology. With the increasing use of LLMs to automate tasks in Knowledge Engineering, the automatic generation of CQs is gaining attention. However, current evaluation approaches lack standardization and reproducibility.

**CQ-Verify** addresses this gap by providing:

## Key Features

- A gold standard dataset derived from real-world ontology engineering projects  
- Multiple evaluation metrics:
  - Cosine Similarity
  - Jaccard Similarity
  - LLM-based semantic analysis (via GPT)
- Visual heatmaps for comparing generated and manually crafted CQs
- Modular and extensible architecture to support additional KE tasks in the future

## Directory Contents

| File / Folder             | Description |
|--------------------------|-------------|
| `app/`                   | Contains the FastAPI application modules and related components. |
| `benchmarkdataset.csv`   | This is the gold standard dataset of manually crafted CQs used for evaluation. |
| `tests/`                 | Directory for test cases and testing utilities. |
| `.gitignore`             | Specifies intentionally untracked files to ignore. |
| `cq_generator_app.py`    | Main entry point for the benchmark system application. |




