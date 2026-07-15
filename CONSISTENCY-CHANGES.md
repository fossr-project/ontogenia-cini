# Bench4KE &mdash; Consistency Update 

Project for Knowledge Engineering Course at UniBo. Semester - Summer 2026.

Authors:
- **Darian Severin Hach**
- **Mikołaj Wewiór**

---


## Overal description

Our work introduces consistency evaluation module. This update is grounded in the **CoLLM** framwork (*'A Framework for Assessing LLM
Consistency in Knowledge Engineering', M. J. Saeedizade et al.*)


### Motivation

Previously, Bench4KE validated generated Competency Questions (CQs) through a single-run pipeline.
Because Large Language Models (LLMs) are inherently non-deterministic and are frequently updated, deprecated, or replaced, a single validation score is insufficient.
A one-off score fails to distinguish whether the result reflects the actual quality of the knowledge engineering method or merely run-to-run stochastic noise.


### Core Solution

To bridge this gap, a new `/consistency/` endpoint has been introduced.
It automates the process of repeating the full generation-to-validation cycle *n* times (with a minimum of *3* runs).
This module fully implements the three core research tests defined in the CoLLM methodology:
- LLM **Repeatability** Test (*T_rep*):
Runs the identical model n times to measure inherent stochasticity and non-determinism.
- LLM **Update Impact** Test (*T_upi*):
Evaluates the effect of changing to a different version within the same model family.
- LLM **Replacement** Test (*T_rpl*):
Assesses the impact of completely swapping the model for an alternative one from a different family.

### Metrics and Reporting

Instead of outputting a single validation score, the system now aggregates results from multiple runs and provides robust statistical and qualitative metrics:
- Mean (μ) and Standard Deviation (σ):
Calculated across the *n* runs for metrics such as BERTScore, BLEU, and ROUGE.
- Performance Drift (PD):
The headline metric comparing the *n*-run mean against the original published baseline score (r_Orig) to indicate performance improvement or decline.
- Qualitative Pass/Fail Verdict:
An automated evaluation (defaulting to a threshold of >= *&ndash;0.05*) that answers whether the original research conclusion still holds despite model updates. 
- Coefficient of Variation (CV):
Used to assign scale-aware stability labels to the results, such as stable, minor variation, or inconsistent.
- Visualizations:
Automatically generates plots to visualize run-to-run drift and variance.


### Architecture Additions

The system architecture has been expanded to support [**OpenRouter**](https://openrouter.ai/) as a new LLM provider, making it easy to swap between various models.
The consistency evaluation operates as a background task via a submit-and-poll mechanism.
Furthermore, the single-run pipeline was extracted into a shared external module to allow code reusability across the project, keeping the original `/validate/` endpoint completely unaffected and backward-compatible. 


---

## How to run

- Create a virtual enviroment and install dependencies from requirements.txt 
- Need 3 terminals, and in each, run:
    ```bash
    source .venv/bin/activate
    export OPENROUTER_API_KEY=sk-or-v1-...
    ```
- Then in the following terminals":
    1. Launch Bench4KE:
        ```bash
        cd ./restapi
        uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
        ```

    1. Launch external cq generator, e.g. Bench4KE built-in one:
        ```bash
        cd ./restapi
        python3 cq_generator_app.py 
        ```

    1. Run consistency evaluation:
        ```bash
        cd ./restapi
        python3 consistency_runner.py --submit --server-url http://127.0.0.1:8000 --config conf/consistency.yaml
        ```

---

## Changes - per functionality

Listed functionality and areas that have been changed.

1. ### Consistency evaluation

    Core functionality of this project.

    #### Files included
    1. new
        - `consistency.py`
        - `consistency_runner.py`
        - `consistency_evaluator.py`
        - `consistency_reporter.py`

    1. modified
        - `main.py`
        - `models.py`
        - `validation_pipeline.py`

1. ### Consistency evaluation configuration 

    Setting up the configuration and parameters for running the evaluation scripts.
    
    ### Files included

    1. new
        - `consistency_configuration.py`
        - `consistency.yaml`
    
    1. modified
        - `models.py`


1. ### New LLM provider &ndash; openrouter

    We added wrappers to use [Openrouter.ai](https://openrouter.ai/).
    It allows to use many of models from different authors easily, by many ways, such as OpenAI SDK.

    #### Files included

    1. modified
        - `cq_generator_app.py`
        - `cq_validator.py`
        - `hit_rate_evaluator.py`
        - `llm_clients.py`

1. ### Reusage of functions moved to external file

    Moved `run_validation_pipeline` outside of `cq_validation.py` to a module outside (`validation_pipeline.py`) that can be imported in other scripts.

    #### Files included

    1. new
        - `validation_pipeline.py`

    1. modified
        - `cq_validation.py`

---


## Changes - per file

Listed files with change description


### [*new*] consistency_runner.py
[./restapi/]

- CLI script to run the consistency evaluation compact and easily.


### [*new*] consistency.py
[./restapi/app/routers]

- A router to run the consistency evaluation with Bench4KE. Works on the same logic as `cq_validation.py`, but calls `consistency_evaluator.py` script.
- Separates complex consistency request to smaller subtasks, which are single Bench4KE validation runs.
- POST /consistency/ (submit) + GET /consistency/{id} (poll).


### [*new*] consistency_evaluator.py
[./restapi/app/services]

- key component covering: reading and loading configuration, collecting tasks, running the task N times (generation of CQs (allowed in parallel), validation run, evaluation) and aggregating the results.
- Both CQs Generation and Validation are prepared to run in parallel. As generation has no issues, then validation can meet sometimes HF/PyTorch errors related to reading/loading same resources while accessing then concurently. We runned the tests with `max_generation_workers` set to 15, and `max_validation_workers` set to 1. 


### [*new*] consistency_reporter.py
[./restapi/app/services]

- Calculates statistics as written in *CoLLM* paper (mean, std, cv, drift).
- Examines if test is stable/consistent, minor variation, or unstable/inconsistent.
- Plots the results and saves them to the file.




### [*new*] consistency_config.py
[./restapi/app/services]

- Works as a script and class to resolve configuration and parameters properly. Written for Hydra-style configuration.


### [*new*] consistency.yaml
[./restapi/conf/]

- Baseline config file to configure key parameters for the consistency evaluation, such as: *n_runs*, *models*, *collm tests*, and others.


### main.py
[./restapi/app/]

- Added router to `consistency` service.


### models.py
[./restapi/app/]

- Added new classes for setting up the configuration.
- Added new classes for customized Responses related to consistency router.






### cq_generator_app.py
[./restapi/]

- Openrouter wrapper added (new llm provider)
    - Exponential backoffs implemented when occured errors during generation
    - Added verification if selected models support temperature selection  
- Minor fixes and features: logger instances added, missing exceptions added

We also founded that way of generating CQs may be improved. We see that for each **single** question in selected project this app calls LLM with same prompt to generate **all** questions. So there is a lot of redundacy which is costly (each llm call costs real money) and takes some time.


### cq_validator.py
[./restapi/app/services]

- Added `provider` attribute in CQValidator class and in `get_llm_client` functino


### hit_rate_evaluator.py
[./restapi/app/services]

- Added `provider` parameter in `_call_llm` function and other related.


### llm_clients.py
[./restapi/app/utils]

- Added Openrouter wrapper for llm calls to this provider. It uses OpenAI SDK (via OpenAIAdapter class)





### [*new*] validation_pipeline.py
[./restapi/app/services]

- Relocated the `run_validation_pipeline` (single-run pipeline) from cq_validation to this file for sharing the pipeline and allowing reusability of the code.
- Added double-checked locking (for parallel validation runs)


### cq_validation.py
[./restapi/app/routers/]

- Moved `run_validation_pipeline` and `normalize_cq_columns` functions outside of a script (to `services` directory), to be able to use it also from other files.
- Imports relocated pipeline.



### requirements.txt
[./]

- updated libraries versions for `python-3.10`

---
