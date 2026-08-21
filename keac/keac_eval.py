"""
KEAC-track evaluator: scores any system's generated Competency Questions against
the full Bench4KE benchmark, using only the AskCQ-derived KEAC metrics
(Coverage, Prec_MMS, ACD, Verbosity Penalty, Final Score) from keac_evaluator.py.

Gold and system rows are grouped by the real input signal(s) each row actually
has — `dataset` (real Dataset content), `scenario` (real Scenario text),
`ontology`/`pdf` (a Link classified by file extension) — and combinations of
these, e.g. "scenario+ontology". This avoids the (Project Name, Name) grouping
bug where multiple distinct personas share a blank Name (e.g. Polifonia) and get
wrongly lumped together. `Description`/`Ontology Description` are deliberately
never used: verified noise for every project that has them.

Input file must have the same columns as cq-genesis output: Project Name, Name,
Scenario, Dataset, Generated CQs (one row per generated CQ). CSV delimiter
(comma or semicolon) is auto-detected.

Use --modes to declare a submission's scope (e.g. --modes scenario,dataset) so
coverage is reported against only the modes actually attempted.

Usage:
    python keac/keac_eval.py <path/to/system_output.csv> [--modes scenario,dataset] [--threshold 0.75] [--out results.csv]
"""

import argparse
import csv
import io
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESTAPI_DIR = os.path.join(REPO_ROOT, "restapi")
sys.path.insert(0, RESTAPI_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.keac_evaluator import (
    compute_coverage,
    compute_mean_maximum_similarity,
    compute_average_centroid_distance,
    compute_verbosity_penalty,
)

from cache_utils import EmbeddingCache

GOLD_CSV = os.path.join(REPO_ROOT, "restapi", "app", "benchmarkdataset.csv")
REAL_DATASET_MIN_LEN = 20


def sniff_delimiter(path: str) -> str:
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
    comma_cols = next(csv.reader(io.StringIO(header), delimiter=","))
    semi_cols = next(csv.reader(io.StringIO(header), delimiter=";"))
    expected = {"Project Name", "Name"}
    if expected.issubset(set(c.strip() for c in semi_cols)):
        return ";"
    if expected.issubset(set(c.strip() for c in comma_cols)):
        return ","
    return ";" if len(semi_cols) > len(comma_cols) else ","


def read_csv_any_delimiter(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delimiter=sniff_delimiter(path), encoding="utf-8-sig")


def is_real_dataset_value(val) -> bool:
    if pd.isna(val):
        return False
    return len(str(val).strip()) > REAL_DATASET_MIN_LEN


def classify_link(url) -> str:
    """Mirrors CQValidator._classify_link in cq_validator.py, kept local so
    keac_eval.py has no dependency on the bench4ke-judge track."""
    if not isinstance(url, str) or not url.strip():
        return "other"
    u = url.strip().lower()
    if any(u.endswith(ext) for ext in (".owl", ".ttl", ".rdf", ".xml", ".n3", ".nt")):
        return "ontology"
    if u.endswith(".pdf") or "pdf" in u:
        return "pdf"
    return "other"


def row_signals(row) -> dict:
    """Returns {signal_name: value} for every real input signal a row actually
    has — a row can carry more than one (e.g. Polifonia/Music Meta Ontology has
    both a real Scenario and a Link to musicmeta.owl -> {"scenario": ..., "ontology": ...}).
    Checked in a fixed order so the combined mode label is deterministic.
    `Description` / `Ontology Description` are deliberately excluded: verified
    noise for every project that has them (gold-CQ compilation notes, or a
    leftover prompt template from generating the gold set itself), not real
    usage-scenario input."""
    signals = {}

    dataset_val = row.get("Dataset")
    if is_real_dataset_value(dataset_val):
        signals["dataset"] = str(dataset_val).strip()

    scenario_val = row.get("Scenario")
    if pd.notna(scenario_val) and str(scenario_val).strip():
        signals["scenario"] = str(scenario_val).strip()

    link_val = row.get("Link")
    link_mode = classify_link(link_val)
    if link_mode in ("ontology", "pdf"):
        signals[link_mode] = str(link_val).strip()

    return signals


def build_groups(df: pd.DataFrame, cq_column: str) -> dict:
    """Returns {(project, match_key): {"mode": mode_label, "cqs": [...]}}.

    `mode_label` is a '+'-joined combination of every real signal a row has
    (e.g. "scenario", "dataset", "scenario+ontology") — informative, shown in
    output. `match_key` is deliberately just the *first* signal value (priority:
    dataset > scenario > ontology > pdf, the order row_signals inserts them),
    not the full combination — a system's output file often only echoes back
    the core prompt content (Scenario/Dataset) and drops supplementary columns
    like Link, so requiring every signal to match would silently orphan groups
    like Polifonia/Music Meta Ontology (gold: scenario+ontology, most tool
    outputs: scenario only) even though they're clearly the same input."""
    groups = {}
    for project_name, proj_df in df.groupby("Project Name"):
        proj_df = proj_df.copy()
        sigs = proj_df.apply(row_signals, axis=1)
        proj_df["_mode"] = sigs.apply(lambda s: "+".join(s.keys()) if s else None)
        proj_df["_match_key"] = sigs.apply(lambda s: next(iter(s.values())) if s else None)

        routable = proj_df[proj_df["_mode"].notna()]
        for key_val, sub_df in routable.groupby("_match_key"):
            cqs = [c for c in sub_df[cq_column].dropna().astype(str).str.strip() if c]
            if not cqs:
                continue
            modes = set(sub_df["_mode"])
            groups[(project_name, key_val)] = {"mode": "+".join(sorted(modes)), "cqs": cqs}

    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", help="System output CSV: Project Name, Name, Scenario, Dataset, Generated CQs")
    parser.add_argument("--gold", default=GOLD_CSV, help="Gold benchmark CSV (default: full benchmarkdataset.csv)")
    parser.add_argument("--threshold", type=float, default=0.75, help="KEAC coverage similarity threshold")
    parser.add_argument("--out", default=None, help="Output CSV path (default: alongside input file)")
    parser.add_argument("--modes", default=None,
                         help="Comma-separated list of modes to score (e.g. 'scenario,dataset'). "
                              "Declares the submission's scope: coverage is reported out of only these "
                              "modes' groups, not the whole benchmark. Default: all modes.")
    args = parser.parse_args()
    declared_modes = {m.strip() for m in args.modes.split(",")} if args.modes else None

    gold_df = read_csv_any_delimiter(args.gold)
    tool_df = read_csv_any_delimiter(args.input_file)

    gold_groups = build_groups(gold_df, "Competency Question")
    tool_groups = build_groups(tool_df, "Generated CQs")

    if declared_modes is not None:
        unknown = declared_modes - {g["mode"] for g in gold_groups.values()}
        if unknown:
            print(f"Warning: --modes has values not present in the gold benchmark: {sorted(unknown)}", file=sys.stderr)
        gold_groups = {k: v for k, v in gold_groups.items() if v["mode"] in declared_modes}

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_cache = EmbeddingCache(model)

    rows = []
    for key, gold_entry in gold_groups.items():
        if key not in tool_groups:
            continue
        gold_cqs = gold_entry["cqs"]
        tool_cqs = tool_groups[key]["cqs"]
        project, key_val = key
        mode = gold_entry["mode"]

        embeddings = embed_cache.encode(tool_cqs + gold_cqs)
        gen_emb, gt_emb = embeddings[: len(tool_cqs)], embeddings[len(tool_cqs):]

        cov = compute_coverage(gen_emb, gt_emb, args.threshold)
        prec_mms = compute_mean_maximum_similarity(gen_emb, gt_emb)
        acd = compute_average_centroid_distance(gen_emb)
        vp = compute_verbosity_penalty(len(tool_cqs), len(gold_cqs))
        harmonic = 0.0 if (cov + prec_mms) == 0 else 2 * (cov * prec_mms) / (cov + prec_mms)
        final_score = harmonic * (1 + 0.1 * acd) * vp

        rows.append({
            "project": project,
            "mode": mode,
            "group_key": str(key_val)[:80],
            "n_gold": len(gold_cqs),
            "n_generated": len(tool_cqs),
            "keac_coverage": round(cov, 4),
            "keac_prec_mms": round(prec_mms, 4),
            "keac_acd": round(acd, 4),
            "keac_verbosity_penalty": round(vp, 4),
            "keac_final_score": round(final_score, 4),
        })

    embed_cache.flush()

    unmatched = len(gold_groups) - len(rows)
    print(f"Matched {len(rows)}/{len(gold_groups)} gold groups "
          f"({unmatched} gold groups had no corresponding rows in {args.input_file}).", file=sys.stderr)

    if not rows:
        print("No groups matched — nothing to score.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.input_file)), "keac_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_gt_total = sum(r["n_gold"] for r in rows)
    weighted_final = sum(r["keac_final_score"] * r["n_gold"] for r in rows) / n_gt_total
    unweighted_final = sum(r["keac_final_score"] for r in rows) / len(rows)
    print(f"\nWrote {len(rows)} group rows to {out_path}")
    print(f"Overall final_score: unweighted mean {unweighted_final:.4f}, gold-size-weighted mean {weighted_final:.4f}")
    print(f"Benchmark coverage: {len(rows)}/{len(gold_groups)} groups scored"
          + (f" (scope declared via --modes {sorted(declared_modes)})" if declared_modes else ""))

    print("\nPer-mode breakdown:")
    total_by_mode = {}
    for g in gold_groups.values():
        total_by_mode[g["mode"]] = total_by_mode.get(g["mode"], 0) + 1
    matched_rows_by_mode = {}
    for r in rows:
        matched_rows_by_mode.setdefault(r["mode"], []).append(r)
    for mode in sorted(total_by_mode):
        total = total_by_mode[mode]
        matched = matched_rows_by_mode.get(mode, [])
        if matched:
            mean_score = sum(r["keac_final_score"] for r in matched) / len(matched)
            print(f"  {mode:<20} {len(matched)}/{total} groups covered, mean final_score {mean_score:.4f}")
        else:
            print(f"  {mode:<20} 0/{total} groups covered (not attempted)")


if __name__ == "__main__":
    main()
