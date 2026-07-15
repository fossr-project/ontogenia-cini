"""
Relocated from `app/routers/cq_validation.py` so both the `/validate/` router
and the `ConsistencyEvaluator` import the same logic. Behavior is unchanged 
for existing callers; an optional `provider` argument is threaded to the 
validation/judge LLMs so the consistency feature can route them through any provider 
e.g., OpenRouter.
"""


import os
import time
import math
import json
import csv
import re
import logging
from typing import Optional

import pandas as pd

import threading

from app.services.cq_validator import CQValidator
from app.services.hit_rate_evaluator import HitRateEvaluator
from app.config import RESULTS_DIR

logger = logging.getLogger(__name__)

_HF_LOAD_LOCK = threading.Lock()
_HF_IS_LOADED = False


def clean_nans(obj):
    if isinstance(obj, list):
        return [clean_nans(o) for o in obj]
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _write_results_csv(results: list, path: str) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list({k for r in results for k in r.keys()})
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# Column names recognised across the CQ pipeline.
_CQ_COLUMN_ALIASES = ("gold standard", "generated", "competency question")


def normalize_cq_columns(df: pd.DataFrame):
    """Lowercase the recognised CQ columns and return (df, gold_col).

    ``gold_col`` is ``"gold standard"`` / ``"competency question"`` / ``None``, if
    neither is present.
    """
    df.columns = [
        c.strip().lower() if c.strip().lower() in _CQ_COLUMN_ALIASES else c
        for c in df.columns
    ]
    if "gold standard" in df.columns:
        return df, "gold standard"
    if "competency question" in df.columns:
        return df, "competency question"
    return df, None


def run_validation_pipeline(
    df: pd.DataFrame,
    gold_col: str,
    output_folder: str,
    model: str,
    validation_mode: str,
    save_results: bool,
    save_every: int,
    evaluator_llm: str,
    tool_llm: str,
    provider: Optional[str] = None,
) -> dict:
    validator = CQValidator(
        output_folder=output_folder, model=model, validation_mode=validation_mode,
        provider=provider,
    )
    results = []
    save_interval = save_every if save_every and save_every > 0 else None
    timestamp = int(time.time())
    results_file = ""
    projects_dir = ""

    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_file = os.path.join(RESULTS_DIR, f"validation_results_{timestamp}.csv")

    project_col = "Project Name" if "Project Name" in df.columns else None

    total_rows = len(df)
    pipeline_start = time.time()
    
    global _HF_IS_LOADED

    for idx, row in df.iterrows():
        base = {"Gold Standard": row[gold_col], "Generated": row["generated"]}
        if project_col:
            base["Project Name"] = row[project_col]
        t0 = time.time()

        try:
            # Double-Checked Locking
            if not _HF_IS_LOADED:
                with _HF_LOAD_LOCK:
                    if not _HF_IS_LOADED:
                        result = validator.validate(row[gold_col], row["generated"])
                        _HF_IS_LOADED = True
                    else:
                        result = validator.validate(row[gold_col], row["generated"])
            else:
                result = validator.validate(row[gold_col], row["generated"])
                
            results.append({**base, **result})
        except Exception as e:
            results.append({**base, "Error": str(e)})

        elapsed = time.time() - t0
        done = idx + 1
        avg = (time.time() - pipeline_start) / done
        remaining = avg * (total_rows - done)
        logger.info("Validated row %s/%s in %.1fs | avg %.1fs | est. remaining %.0fm%.0fs",
                    done, total_rows, elapsed, avg, remaining // 60, remaining % 60)

        if save_results and save_interval and len(results) % save_interval == 0:
            _write_results_csv(results, results_file)
            logger.info("Incremental save after %s rows → %s", len(results), results_file)

    if save_results:
        _write_results_csv(results, results_file)

        # Save one summary CSV with one row per project
        if project_col and results:
            _NUMERIC_COLS = [
                "Average Cosine Similarity", "Max Cosine Similarity",
                "Average BERTScore-F1", "Max BERTScore-F1",
                "Average BERTScore-Precision", "Average BERTScore-Recall",
                "Average BLEU", "Max BLEU",
                "Average ROUGE-L F1", "Max ROUGE-L F1",
                "Average Jaccard Similarity",
                "Precision@0.6", "Matches@0.6",
            ]
            project_buckets: dict = {}
            for r in results:
                pname = str(r.get("Project Name", "unknown")).strip() or "unknown"
                project_buckets.setdefault(pname, []).append(r)
            summary_rows = []
            for pname, rows in project_buckets.items():
                row_out = {"Project Name": pname, "Num Rows": len(rows)}
                for col in _NUMERIC_COLS:
                    vals = [r[col] for r in rows if isinstance(r.get(col), (int, float)) and r[col] == r[col]]
                    row_out[col] = round(sum(vals) / len(vals), 4) if vals else None
                summary_rows.append(row_out)
            projects_dir = os.path.join(RESULTS_DIR, f"scores_by_project_{timestamp}.csv")
            _write_results_csv(summary_rows, projects_dir)
            logger.info("Per-project summary CSV saved to %s", projects_dir)

        try:
            grouped = validator.aggregate_dataframe_metrics(
                df, gold_col=gold_col, generated_col="generated",
                link_cols=["Link"],
                output_ontologies_folder=os.path.join(RESULTS_DIR, f"downloaded_ontologies_{timestamp}"),
                precision_threshold=0.6,
                compute_pairwise_metrics=True,
            )
            with open(os.path.join(RESULTS_DIR, f"validation_grouped_{timestamp}.json"), "w", encoding="utf-8") as gf:
                json.dump(grouped, gf, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Aggregate metrics failed: %s", e)

    # Hit Rate metric
    hit_rate_result = {}
    try:
        def split_cqs(text: str) -> list:
            return [q.strip() + "?" for q in re.split(r"\?", str(text)) if q.strip()]

        bench_cqs, tool_cqs = [], []
        for _, row in df.iterrows():
            bench_cqs.extend(split_cqs(row[gold_col]))
            tool_cqs.extend(split_cqs(row["generated"]))

        context_cols = [c for c in df.columns if c in ("Scenario", "Description")]
        scenario_context = " ".join(
            str(df[col].dropna().iloc[0]) for col in context_cols if not df[col].dropna().empty
        ) or "No scenario context provided."

        hit_rate_result = HitRateEvaluator(threshold=0.6, k=3).compute(
            bench_cqs=bench_cqs,
            tool_cqs=tool_cqs,
            scenario_context=scenario_context,
            evaluator_llm=evaluator_llm,
            tool_llm=tool_llm,
            provider=provider,
        )
    except Exception as e:
        hit_rate_result = {"error": str(e)}

    # Always save hit rate to disk so it's not lost if the HTTP connection times out
    if save_results and results_file:
        hit_rate_path = results_file.replace("validation_results_", "hit_rate_")
        hit_rate_path = os.path.splitext(hit_rate_path)[0] + ".json"
        try:
            with open(hit_rate_path, "w", encoding="utf-8") as hf:
                json.dump(clean_nans(hit_rate_result), hf, ensure_ascii=False, indent=2)
            logger.info("Hit rate saved to %s", hit_rate_path)
        except Exception as e:
            logger.warning("Could not save hit rate to disk: %s", e)

    return {
        "message": "Processing complete",
        "results_saved_to": results_file if save_results else "Not saved",
        "per_project_scores": projects_dir if projects_dir else "Not saved",
        "validation_results": clean_nans(results),
        "hit_rate": clean_nans(hit_rate_result),
    }
