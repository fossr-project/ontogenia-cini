"""Configuration resolution for the ConsistencyEvaluator.

1. **Baseline config file** for reproducible, version-controlled setup — a
   YAML/JSON file with the nested ``generation`` / ``evaluation`` / ``dataset`` /
   ``tests`` shape (see ``conf/consistency.yaml``).
2. **Single-parameter overrides** — Dotted ``key.path=value`` tokens
   (e.g. ``evaluation.n_runs=3``,
   ``tests.repeatability.models=[openrouter/openai/gpt-4o]``) layered on top.

``resolve_config(file_dict, overrides)`` returns a validated ``ConsistencyConfig``.
"""
from __future__ import annotations

import ast
import copy
import json
from typing import Any, Dict, List, Optional

from app.models import ConsistencyConfig


# Nested base config (mirrors the shape of conf/consistency.yaml).
DEFAULT_BASE: Dict[str, Any] = {
    "generation": {
        "provider": "openai",
        "temperature": None,   # forwarded to the generator only where the API supports it
        "service_url": "http://127.0.0.1:8001/newapi",
    },
    "evaluation": {
        "n_runs": 3,
        "validation_mode": "all",
        "validation_llm_provider": "openai",
        "validation_model": "gpt-4",
        "evaluator_llm": "gpt-4",
        "tool_llm": None,
        "output_dir": "consistency_results",
        "save_results": False,
        "poll_interval": 10,
        "max_generation_workers": 15,
        "max_validation_workers": 1,
        "original_scores": None,    # {metric_name: r_Orig} for Performance Drift / pass-fail
        "pass_threshold": -0.05,    # pass if PD >= this
        "recompute_baseline": True, # fill missing baseline/r_Orig from the original model's repeatability mean
    },
    "dataset": {
        "gold_standard_path": None,
        "projects": None,
        "max_rows": None,
    },
    "tests": {
        "repeatability": {"enabled": True, "models": []},
        "update_impact": {"enabled": True, "models": []},
        "replacement": {"enabled": True, "models": []},
    },
}


def _coerce(value: str) -> Any:
    """
    Auxiliary method to coerce a raw override string to a Python scalar/list.

    Handles null/none, true/false, ints, floats, ``[a, b, c]`` lists (quoted or
    bare items), and otherwise returns the raw string. Only *flat* lists are
    supported — nested brackets (``[[a],[b]]``) are not.
    """
    v = value.strip()
    low = v.lower()
    if low in ("null", "none", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False

    # Bracketed list or quoted string: try a literal eval first.
    if (v.startswith("[") and v.endswith("]")) or (v[:1] in ("'", '"')):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            if v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                if not inner:
                    return []
                return [_coerce(item) for item in inner.split(",")]
            return v

    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _set_dotted(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = target
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value


def apply_overrides(base: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Merge Hydra-style dotted ``key.path=value`` overrides onto a deep copy of ``base``."""
    result = copy.deepcopy(base)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override (expected key=value): {ov!r}")
        key, _, raw = ov.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override (empty key): {ov!r}")
        _set_dotted(result, key, _coerce(raw))
    return result


def build_config(
    overrides: Optional[List[str]] = None,
    base: Optional[Dict[str, Any]] = None,
) -> ConsistencyConfig:
    """Resolve overrides against the base config and validate as ``ConsistencyConfig``."""
    cfg = apply_overrides(base if base is not None else DEFAULT_BASE, overrides or [])
    gen = cfg["generation"]
    ev = cfg["evaluation"]
    ds = cfg["dataset"]
    tests = cfg["tests"]
    return ConsistencyConfig(
        n_runs=ev["n_runs"],
        external_service_url=gen["service_url"],
        generator_llm_provider=gen["provider"],
        generator_temperature=gen["temperature"],
        validation_mode=ev["validation_mode"],
        validation_llm_provider=ev["validation_llm_provider"],
        validation_model=ev["validation_model"],
        evaluator_llm=ev["evaluator_llm"],
        tool_llm=ev["tool_llm"],
        gold_standard_path=ds["gold_standard_path"],
        dataset_projects=ds["projects"],
        dataset_max_rows=ds["max_rows"],
        output_dir=ev["output_dir"],
        save_results=ev["save_results"],
        poll_interval=ev["poll_interval"],
        max_generation_workers=ev["max_generation_workers"],
        max_validation_workers=ev["max_validation_workers"],
        original_scores=ev["original_scores"],
        pass_threshold=ev["pass_threshold"],
        recompute_baseline=ev["recompute_baseline"],
        repeatability=tests["repeatability"],
        update_impact=tests["update_impact"],
        replacement=tests["replacement"],
    )


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a deep copy of ``base``. The override takes precedence."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config_file(content, filename: str = "") -> Dict[str, Any]:
    """Parse a baseline config file (YAML or JSON) into a nested dict.
    YAML is used only for ``.yaml``/``.yml`` filenames, everything else is parsed as JSON.
    """
    text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else content
    if filename.lower().endswith((".yaml", ".yml")):
        import yaml  # optional dependency, present in this environment
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping/object at the top level.")
    return data


def resolve_config(
    file_dict: Optional[Dict[str, Any]] = None,
    overrides: Optional[List[str]] = None,
) -> ConsistencyConfig:
    """Baseline config file (deep-merged onto defaults) + dotted single-param overrides."""
    base = _deep_merge(DEFAULT_BASE, file_dict or {})
    return build_config(overrides=overrides, base=base)
