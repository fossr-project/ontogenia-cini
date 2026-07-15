"""
ConsistencyEvaluator - repeats the generation -> validation cycle ``n`` times.

Generation-agnostic: every run regenerates via ``call_external_cq_generation_service``, 
then validates and scores through the shared ``run_validation_pipeline``.
"""


import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from concurrent.futures import Future, ThreadPoolExecutor, wait
import threading

from app.config import DEFAULT_DATASET
from app.utils.external_call import call_external_cq_generation_service
from app.services.validation_pipeline import run_validation_pipeline, normalize_cq_columns
from app.services import consistency_reporter as reporter
from app.models import ConsistencyConfig

logger = logging.getLogger(__name__)

# Per-row numeric metrics emitted by CQValidator.validate() that get averaged per run.
_NUMERIC_OVERALL_KEYS = [
    "Average Cosine Similarity", "Max Cosine Similarity",
    "Average Jaccard Similarity",
    "Average BERTScore-F1", "Max BERTScore-F1",
    "Average BERTScore-Precision", "Average BERTScore-Recall",
    "Average BLEU", "Max BLEU",
    "Average ROUGE-L F1", "Max ROUGE-L F1",
    "Precision@0.6", "Matches@0.6",
]


class ConsistencyEvaluator:
    def __init__(self, config: ConsistencyConfig,
                 progress_callback: Optional[Callable[[dict], None]] = None):
        self.config = config
        self._progress = progress_callback

        self._active_workers = 0
        self._completed_tasks = 0
        self._max_generation_workers = getattr(self.config, 'max_generation_workers', 6) or 6
        self._max_validation_workers = getattr(self.config, 'max_validation_workers', 1) or 1
        self._lock = threading.Lock()

    def _emit(self, event: dict) -> None:
        if self._progress:
            try:
                self._progress(event)
            except Exception as e:
                logger.warning("progress_callback raised: %s", e)

    def _load_dataset(self) -> Tuple[pd.DataFrame, str]:
        """Load the dataset from `cfg.standard_path` or `DEFAULT_DATASET`"""
        cfg = self.config
        df = pd.read_csv(cfg.gold_standard_path or DEFAULT_DATASET)
        # Shared CQ-column normalization + gold-column detection (same as /validate/).
        df, gold_col = normalize_cq_columns(df)
        if gold_col is None:
            raise ValueError("Dataset must contain a 'gold standard' or 'Competency Question' column.")

        # Each run regenerates - drop any pre-existing generated column.
        if "generated" in df.columns:
            df = df.drop(columns=["generated"])
        if cfg.dataset_projects and "Project Name" in df.columns:
            wanted = {p.strip() for p in cfg.dataset_projects}
            df = df[df["Project Name"].astype(str).str.strip().isin(wanted)]
        if cfg.dataset_max_rows is not None:
            df = df.head(cfg.dataset_max_rows)
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError("Dataset is empty after filtering.")
        return df, gold_col

    def _generate_cqs(self, df: pd.DataFrame, gen_model: Optional[str]) -> pd.DataFrame:
        """Run CQ generation via external tool / API call."""
        cfg = self.config
        df_run = df.copy()
        return call_external_cq_generation_service(
            df_run,
            cfg.external_service_url,
            llm_provider=cfg.generator_llm_provider,
            model=gen_model,
        )

    def _validate_cqs(self, df_run: pd.DataFrame, gold_col: str) -> dict:
        """Run validation locally."""
        cfg = self.config
        result = run_validation_pipeline(
            df=df_run,
            gold_col=gold_col,
            output_folder=os.path.join(cfg.output_dir, "heatmaps"),
            model=cfg.validation_model,
            validation_mode=cfg.validation_mode,
            save_results=cfg.save_results,
            save_every=1,
            evaluator_llm=cfg.evaluator_llm,
            tool_llm=cfg.tool_llm,
            provider=cfg.validation_llm_provider,
        )
        return self._extract_overall_metrics(result)

    @staticmethod
    def _extract_overall_metrics(result: dict) -> Dict[str, float]:
        rows = result.get("validation_results") or []
        metrics: Dict[str, float] = {}
        for key in _NUMERIC_OVERALL_KEYS:
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                metrics[key] = float(sum(vals) / len(vals))
        if rows:
            errored = sum(1 for r in rows if "Error" in r)
            metrics["error_rate"] = errored / len(rows)
        # Hit-rate coverage - evaluator-stochastic (HitRate runs at temperature 0.7).
        hit_rate = result.get("hit_rate") or {}
        for key in ("coverage_rate", "augmented_coverage_rate"):
            if isinstance(hit_rate.get(key), (int, float)):
                metrics[key] = float(hit_rate[key])
        return metrics

    def _thread_generate_wrapper(self, task: dict, df: pd.DataFrame, total_tasks: int) -> None:
        """Multithread wrapper for generation phase."""
        with self._lock:
            self._active_workers += 1
            current_active = self._active_workers
            current_completed = self._completed_tasks

        self._emit({
            "status": "running",
            "test": task["test"],
            "model": task["model"],
            "run": task["run_idx"],
            "total_runs": self.config.n_runs, 
            "phase": "Generation (API)",
            "active_workers": current_active,
            "max_workers": self._max_generation_workers,
            "completed_tasks": current_completed,
            "total_tasks": total_tasks
        })

        try:
            task["generated_df"] = self._generate_cqs(df, task["model"])
        except Exception as e:
            logger.error(f"Generation failed for {task['test']} / {task['model']}: {e}")
            task["generated_df"] = None
        finally:
            with self._lock:
                self._active_workers -= 1
                self._completed_tasks += 1
                current_completed = self._completed_tasks
        self._emit({
            "status": "running",
            "test": task["test"],
            "model": task["model"],
            "run": task["run_idx"],
            "total_runs": self.config.n_runs, 
            "phase": "Generation (API)",
            "active_workers": current_active,
            "max_workers": self._max_generation_workers,
            "completed_tasks": current_completed,
            "total_tasks": total_tasks
        })

    def _thread_validate_wrapper(self, task: dict, gold_col: str, i: int, total_tasks: int) -> None:
        with self._lock:
            self._active_workers += 1
            current_active = self._active_workers
            current_completed = self._completed_tasks
        
        self._emit({
            "status": "running",
            "test": task["test"],
            "model": task["model"],
            "run": task["run_idx"],
            "total_runs": self.config.n_runs, 
            "phase": "Validation (local computing)",
            "active_workers": current_active,
            "max_workers": self._max_validation_workers,
            "completed_tasks": current_completed or 0,
            "total_tasks": total_tasks
        })

        try:
            if task.get("generated_df") is not None:
                try:
                    task["result"] = self._validate_cqs(task["generated_df"], gold_col)
                except Exception as e:
                    logger.error(f"Validation crashed for {task['test']} / {task['model']}: {e}")
                    task["result"] = {}
            else:
                logger.error(f"Missing generated dataframe for {task['test']} - {task['model']}")
                task["result"] = {}
        finally:
            with self._lock:
                self._active_workers -= 1
                self._completed_tasks += 1
                current_completed = self._completed_tasks
        self._emit({
            "status": "running",
            "test": task["test"],
            "model": task["model"],
            "run": task["run_idx"],
            "total_runs": self.config.n_runs, 
            "phase": "Validation (local computing)",
            "active_workers": current_active,
            "max_workers": self._max_validation_workers,
            "completed_tasks": current_completed or 0,
            "total_tasks": total_tasks
        })

    @staticmethod
    def _safe(model: str) -> str:
        return str(model).replace("/", "_")

    def _stats(self, runs: List[dict],
               recomputed_scores: Optional[Dict[str, float]] = None) -> Dict[str, dict]:
        """Compute dispersion statistics for the generated results."""
        return reporter.compute_stats(
            runs, self.config.original_scores, self.config.pass_threshold,
            recomputed_scores=recomputed_scores)

    def _calculate_model_stats(
            self,
            runs: List[dict],
            model: Optional[str],
            test_name: str,
            recomputed_scores: Optional[Dict[str, float]] = None
        ) -> dict:
        """Calculates statistics and generates plots for collected results."""
        stats = self._stats(runs, recomputed_scores=recomputed_scores)
        return {
            "model": model,
            "runs": runs,
            "stats": stats,
            "pass_fail": reporter.summarize_pass_fail(stats),
            "plots": reporter.generate_plots(
                stats,
                f"{test_name}_{self._safe(model)}",
                self.config.output_dir
            ),
            "summary": reporter.generate_qualitative_summary(stats, test_name),
        }

    def run_all(self) -> dict:
        """Main orchestration method for evaluation run."""
        cfg = self.config
        df, gold_col = self._load_dataset()
        result: Dict[str, object] = {
            "config": json.loads(cfg.json()),
            "dataset_rows": len(df),
            "notes": [],
        }

        # Phase 0: collecting all tasks requirements
        all_tasks = []
        n = cfg.n_runs

        if cfg.repeatability.enabled and cfg.repeatability.models:
            rep_model = cfg.repeatability.models[0]
            for i in range(n):
                all_tasks.append({"test": "repeatability", "model": rep_model, "run_idx": i + 1})
        else:
            logger.warning("No base model enabled. Quiting.")
            return result

        if cfg.update_impact.enabled and cfg.update_impact.models:
            for model in cfg.update_impact.models:
                for i in range(n):
                    all_tasks.append({"test": "update_impact", "model": model, "run_idx": i + 1})

        if cfg.replacement.enabled and cfg.replacement.models:
            for model in cfg.replacement.models:
                for i in range(n):
                    all_tasks.append({"test": "replacement", "model": model, "run_idx": i + 1})

        total_tasks = len(all_tasks)
        if total_tasks == 0:
            logger.warning("No tests enabled or no models specified.")
            return result

        logger.info(f"Planned {total_tasks} total generation tasks. Max workers: {self._max_generation_workers}")

        # Phase 1: generating CQs
        self._completed_tasks = 0
        with ThreadPoolExecutor(max_workers=self._max_generation_workers) as executor:
            futures = []
            for task in all_tasks:
                futures.append(executor.submit(self._thread_generate_wrapper, task, df, total_tasks))
            # Block the code, wait until generation tasks are finished
            wait(futures)

        logger.info("Parallel Generation completed. Moving to Validation.")

        # Phase 2: validation
        self._active_workers = 0
        self._completed_tasks = 0
        
        with ThreadPoolExecutor(max_workers=self._max_validation_workers) as executor:
            val_futures = []
            for i, task in enumerate(all_tasks):
                val_futures.append(executor.submit(self._thread_validate_wrapper, task, gold_col, i, total_tasks))
            # Block the code, wait until validation tasks are finished
            wait(val_futures)

        logger.info("Validation completed. Moving to Evaluation")

        # Phase 3: Consistency Evaluation
        recomputed_scores: Optional[Dict[str, float]] = None

        if cfg.repeatability.enabled:
            rep_tasks = [t for t in all_tasks if t["test"] == "repeatability"]
            if rep_tasks:
                model = rep_tasks[0]["model"]
                runs = [t.get("result", {}) for t in rep_tasks]
                rep_result = self._calculate_model_stats(runs, model, "repeatability", None)
                result["repeatability"] = rep_result
                if cfg.recompute_baseline:
                    recomputed_scores = {metric: s["mean"] for metric, s in rep_result["stats"].items()}

        if cfg.update_impact.enabled:
            ui_results = {}
            for model in cfg.update_impact.models:
                model_tasks = [t for t in all_tasks if t["test"] == "update_impact" and t["model"] == model]
                if model_tasks:
                    runs = [t.get("result", {}) for t in model_tasks]
                    ui_results[model] = self._calculate_model_stats(runs, model, "update_impact", recomputed_scores)
            result["update_impact"] = {"by_model": ui_results}

        if cfg.replacement.enabled:
            repl_results = {}
            for model in cfg.replacement.models:
                model_tasks = [t for t in all_tasks if t["test"] == "replacement" and t["model"] == model]
                if model_tasks:
                    runs = [t.get("result", {}) for t in model_tasks]
                    repl_results[model] = self._calculate_model_stats(runs, model, "replacement", recomputed_scores)
            result["replacement"] = {"by_model": repl_results}

        # Phase 4: Save to the file
        os.makedirs(cfg.output_dir, exist_ok=True)
        out_path = os.path.join(cfg.output_dir, f"consistency_report_{int(time.time())}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            result["report_path"] = out_path
        except Exception as e:
            logger.warning("Could not write consistency report: %s", e)
            
        return result
