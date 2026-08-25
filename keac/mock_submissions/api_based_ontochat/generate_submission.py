"""
Calls OntoChat's live public API (b289zhan/OntoChat on Hugging Face Spaces) for
a small subset of scenario-mode gold groups, and writes submission_output.csv
in the schema keac_eval.py expects.

Run from the repo root:
    OPENAI_API_KEY=... python keac/mock_submissions/api_based_ontochat/generate_submission.py

Requires: pip install gradio_client
"""

import csv
import os
import time

import httpx
from gradio_client import Client

csv.field_size_limit(10**7)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GOLD_CSV = os.path.join(REPO_ROOT, "restapi", "app", "benchmarkdataset_full.csv")
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_output.csv")

# Three scenario-mode gold groups sampled for this demo, out of 16 available.
# OntoChat's public API only exposes a scenario/user-story generator, so this
# submission intentionally never touches dataset/pdf/ontology-file groups.
SCENARIO_STARTS = {
    "Amy1": "Amy wants to assess",
    "Paul1": "Paul is hired",
    "Linka": "Persona\nLinka",
}


def load_targets():
    with open(GOLD_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    targets = {}
    for r in rows:
        if r["Project Name"] != "Polifonia":
            continue
        scenario = (r.get("Scenario") or "").strip()
        for label, prefix in SCENARIO_STARTS.items():
            if label not in targets and scenario.startswith(prefix):
                targets[label] = {"scenario": scenario, "name": r.get("Name")}
    missing = set(SCENARIO_STARTS) - set(targets)
    if missing:
        raise RuntimeError(f"Could not find gold rows for: {missing}")
    return targets


def main():
    api_key = os.environ["OPENAI_API_KEY"]
    targets = load_targets()

    client = Client("b289zhan/OntoChat", httpx_kwargs={"timeout": httpx.Timeout(120.0)})
    client.predict(api_key, api_name="/set_openai_api_key")

    history = [[None, "I am OntoChat, your conversational ontology engineering assistant. "
                      "Here is the second step of the system. Please give me your user story "
                      "and tell me how many competency questions you want me to generate from the user story."]]

    rows = []
    for label, entry in targets.items():
        prompt = (
            f'Here is a user scenario for ontology engineering:\n'
            f'"""\n{entry["scenario"]}\n"""\n\n'
            f"Please generate up to five competency questions for this scenario "
            f"and return them on one single line, separated by a semicolon (;)."
        )
        print(f"--- calling OntoChat for {label} ---")
        result = client.predict(prompt, history, api_name="/cq_generator")
        answer = result[0] if isinstance(result, (list, tuple)) else result
        print(answer[:300])
        cqs = [c.strip() for c in answer.split(";") if c.strip()]
        for cq in cqs:
            if not cq.endswith("?"):
                cq = cq + "?"
            rows.append({
                "Project Name": "Polifonia",
                "Name": entry["name"],
                "Scenario": entry["scenario"],
                "Dataset": "",
                "Generated CQs": cq,
            })
        time.sleep(2)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Project Name", "Name", "Scenario", "Dataset", "Generated CQs"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
