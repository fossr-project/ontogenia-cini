"""Statistics, qualitative labels, and base64 plots for the ``ConsistencyEvaluator``.

Pure functions over the per-run metric dicts produced by
``ConsistencyEvaluator._extract_overall_metrics``. Plots reuse the base64-PNG
pattern from ``heatmap_generator.py``.
"""
import base64
import io
import math
import os
import statistics
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt 
import seaborn as sns           


# CV thresholds for qualitative stability labels.
_STABLE_MAX = 0.05
_MINOR_MAX = 0.15


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def _label(std: float, cv) -> str:
    if std == 0:
        return "stable"
    if cv is None:
        return "inconsistent"
    if cv < _STABLE_MAX:
        return "stable"
    if cv < _MINOR_MAX:
        return "minor variation"
    return "inconsistent"


def compute_stats(
    runs: List[Dict[str, float]],
    original_scores: Optional[Dict[str, float]] = None,
    pass_threshold: float = -0.05,
    recomputed_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, dict]:
    """For each metric present in every run, compute dispersion stats.

    If a metric has a reference score in ``r_Orig`` - the published value from
    ``original_scores`` or, when absent, the recomputed value from ``recomputed_scores`` 
    which is the original model's repeatability mean. It also computes the **Performance Drift**
    ``pd = (mean - r_Orig) / r_Orig``, a per-metric ``passed`` flag i.e., ``pd >= pass_threshold``, 
    and ``r_orig_source`` (``"published"`` or ``"recomputed"``). Published scores take precedence per metric.
    """
    if not runs:
        return {}
    keys = set().union(*[set(r.keys()) for r in runs])
    stats: Dict[str, dict] = {}
    for key in sorted(keys):
        vals = [r.get(key) for r in runs]
        if not all(_is_number(v) for v in vals):
            continue  # metric missing/non-numeric in some run - skip
        vals = [float(v) for v in vals]
        mean = sum(vals) / len(vals)
        # Compute std
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        cv = (std / mean) if mean != 0 else None
        drift = [v - vals[0] for v in vals]
        entry = {
            "per_run": vals,
            "mean": mean,
            "std": std,
            "cv": cv,
            "drift": drift,
            "label": _label(std, cv),
        }
        # Published r_Orig takes precedence per metric.
        # Otherwise fall back to the recomputed baseline.
        published = (original_scores or {}).get(key)
        if _is_number(published) and published != 0:
            orig, source = float(published), "published"
        else:
            recomputed = (recomputed_scores or {}).get(key)
            orig, source = (float(recomputed), "recomputed") \
                if _is_number(recomputed) and recomputed != 0 else (None, None)
        if orig is not None:
            pd = (mean - orig) / orig
            entry["r_orig"] = orig
            entry["pd"] = pd    # Performance Drift
            entry["passed"] = pd >= pass_threshold
            entry["r_orig_source"] = source
        stats[key] = entry
    return stats


def summarize_pass_fail(stats: Dict[str, dict]) -> dict:
    evaluated = [s for s in stats.values() if "passed" in s]
    passed = sum(1 for s in evaluated if s["passed"])
    failed = len(evaluated) - passed
    overall = None if not evaluated else ("pass" if failed == 0 else "fail")
    return {"overall": overall, "passed": passed, "failed": failed,
            "evaluated": len(evaluated)}


def generate_plots(stats: Dict[str, dict], test_name: str, output_dir: str) -> Dict[str, str]:
    plots: Dict[str, str] = {}
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for metric, s in stats.items():
        runs_x = list(range(1, len(s["per_run"]) + 1))
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            sns.lineplot(x=runs_x, y=s["per_run"], marker="o", ax=ax)
            ax.axhline(s["mean"], linestyle="--", color="gray", label=f"mean={s['mean']:.3f}")
            ax.set_title(f"{test_name} - {metric} ({s['label']})")
            ax.set_xlabel("run")
            ax.set_ylabel(metric)
            ax.legend()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode("utf-8")
        finally:
            plt.close(fig)
        safe = metric.replace("/", "_").replace(" ", "_")
        with open(os.path.join(plot_dir, f"{test_name}_{safe}.png"), "wb") as f:
            f.write(base64.b64decode(encoded))
        plots[metric] = encoded
    return plots


def generate_qualitative_summary(stats: Dict[str, dict], test_name: str = "") -> str:
    labels = [s["label"] for s in stats.values()]
    total = len(labels)
    stable = labels.count("stable")
    minor = labels.count("minor variation")
    inconsistent = labels.count("inconsistent")
    suffix = f" under {test_name} conditions." if test_name else "."
    return (f"{stable} of {total} metrics are stable, {minor} show minor variation, "
            f"{inconsistent} are inconsistent{suffix}")
