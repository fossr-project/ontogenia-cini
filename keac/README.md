# KEAC track

Tooling to score a system's generated Competency Questions against the
Bench4KE benchmark using the AskCQ-derived metrics from
[`keac_evaluator.py`](../restapi/app/services/keac_evaluator.py) (from
[PR #13](https://github.com/fossr-project/ontogenia-cini/pull/13)): Coverage,
Mean Maximum Similarity (Prec_MMS), Average Centroid Distance (ACD), Verbosity
Penalty, and a combined Final Score. No LLM judge, no API key required to
*score* a submission — only some submission methods (e.g. calling a live
system) need one to *produce* CQs in the first place.

This mirrors the submission process of the [Knowledge Engineering Automation
Challenge](https://codeberg.org/ke-automation-challenge/challenge-catalog)
(KEAC), which accepts three submission modes — see the two worked examples in
[`mock_submissions/`](mock_submissions/).

## Files

| File | Purpose |
|---|---|
| `keac_eval.py` | The scorer. Takes any system's output CSV + the gold benchmark, groups both by real input signal, computes the KEAC metrics per group. |
| `cache_utils.py` | Persistent disk cache for SBERT embeddings (`.cache/`, gitignored) — reruns don't re-embed unchanged text. |
| `mock_submissions/` | Two worked examples of the submission process end to end (see below). |

## If you want to submit to KEAC

A submission is a pull request to the
[challenge-catalog](https://codeberg.org/ke-automation-challenge/challenge-catalog)
repo, following its
[`PULL_REQUEST_TEMPLATE.md`](https://codeberg.org/ke-automation-challenge/challenge-catalog/raw/branch/main/.gitea/PULL_REQUEST_TEMPLATE.md).
That template asks for a **metadata file** alongside your **result file** —
there's no fixed schema mandated yet (the catalog repo is very new), but
`mock_submissions/*/metadata.yaml` in this folder is a reasonable structure to
copy, since its fields map directly onto the PR template's sections
(submission info, task/track, submission mode, system/model info, artifacts,
reproduction instructions).

### 1. Pick your submission mode

The PR template recognises three:

- **Output-based** — you already have a file of generated CQs. See
  [`mock_submissions/output_based_cq_genesis/`](mock_submissions/output_based_cq_genesis/).
- **API-based** — your system is callable live. See
  [`mock_submissions/api_based_ontochat/`](mock_submissions/api_based_ontochat/),
  which actually calls OntoChat's public Hugging Face Space.
- **System-based** — you provide source + instructions to reproduce locally. Not
  demonstrated here yet, but the same result/metadata shape applies.

### 2. Declare your input modality honestly

The gold benchmark has five kinds of groups: `dataset`, `scenario`
(persona/user story), `ontology` (a Link to an ontology file), `pdf` (a Link
to a reference paper), and combinations like `scenario+ontology`. Most
systems only handle some of these — that's expected, not a defect. Use
`--modes` to declare which ones you're attempting:

```bash
python keac/keac_eval.py your_output.csv --modes scenario,dataset
```

Without `--modes`, coverage is reported out of the whole benchmark (67
groups), which will look artificially low for a system that only ever
targeted one input type. **Declare your scope** — see the `selected_track_or_input_modality`
field in the metadata examples.

### 3. Produce your result file

Your output CSV needs these columns: `Project Name`, `Name`, `Scenario`,
`Dataset`, `Generated CQs` — one row per generated question. This is the same
schema the gold benchmark itself uses for `Scenario`/`Dataset` (see
`restapi/app/benchmarkdataset_full.csv`), so a system that consumes a gold row
as input can usually echo those columns straight back.

You don't need to fill in every column — `keac_eval.py` only needs `Project
Name` and `Generated CQs` populated, plus whichever of `Scenario`/`Dataset`
your system's input actually came from (see `mock_submissions/api_based_ontochat/`,
which leaves `Dataset` empty since it's a scenario-only system).

### 4. Score it

```bash
python keac/keac_eval.py <your_output.csv> \
  --gold restapi/app/benchmarkdataset_full.csv \
  --modes <your declared modes> \
  --out keac_results.csv
```

This prints a per-mode coverage breakdown to stderr and writes one row per
matched group to `--out`. Both go into your submission alongside the metadata
file.

### 5. Write the metadata file

Copy one of `mock_submissions/*/metadata.yaml` and fill in your own values —
team info, system/model details, artifacts (repo/API/paper links), and the
exact commands you ran (so the organisers can reproduce your numbers).

## Worked examples

| | `output_based_cq_genesis/` | `api_based_ontochat/` |
|---|---|---|
| Submission mode | Output-based | API-based |
| System | CQ-Genesis (pre-generated file) | OntoChat, called live via `gradio_client` |
| Modes attempted | `dataset`, `scenario` | `scenario` only |

Both are genuinely worked examples, not fabricated numbers, and both hit the
same lesson: **check what a system's input actually was before trusting the
gold group's mode label.** Polifonia/Linka is a `scenario+ontology` gold group
(it has both a persona text and a link to `musicmeta.owl`), but neither
CQ-Genesis's own README nor OntoChat's API expose any ontology-file input —
both only ever received the scenario text. So Linka's output is kept in each
`submission_output.csv` for transparency, but excluded from the scored/declared
scope in both. A real submission would attempt all 16 available scenario
groups (kept small here to limit API cost). Neither is a real KEAC submission.
