from flask import Flask, render_template_string, request, redirect, url_for, flash
import requests, os, json
from dotenv import load_dotenv
load_dotenv()
API_TIMEOUT = int(os.getenv("CQ_API_TIMEOUT", "3600"))
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

API_BASE = os.environ.get("CQ_API_URL", "http://127.0.0.1:8000")
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "changeme")


def _load_json_file(path: str):
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _summarize_oops(oops_obj):
    if not isinstance(oops_obj, dict):
        return oops_obj
    out = {}
    for k in (
        "error",
        "status_code",
        "parse_error",
        "skipped",
        "reason",
        "pitfalls_total",
        "pitfall_codes",
        "pitfall_instance_counts",
        "pitfall_affected_elements_counts",
        "pitfalls",
    ):
        if k in oops_obj:
            out[k] = oops_obj.get(k)
    # Keep only a small preview of raw responses (full raw is already stored on disk).
    raw = oops_obj.get("raw_response")
    if isinstance(raw, str) and raw:
        out["raw_response_preview"] = raw[:2000]
        if len(raw) > 2000:
            out["raw_response_truncated"] = True
    return out


def _summarize_llm_eval(llm_obj):
    if not isinstance(llm_obj, dict):
        return llm_obj
    out = {"summary": llm_obj.get("summary")}
    results = llm_obj.get("results") or []
    if isinstance(results, list):
        trimmed = []
        for r in results:
            if not isinstance(r, dict):
                continue
            trimmed.append(
                {
                    "competency_question": r.get("competency_question"),
                    "label": r.get("label"),
                    "sparql": r.get("sparql"),
                    "error": r.get("error"),
                }
            )
        out["results"] = trimmed
    return out

TEMPLATE = """
<!doctype html>
<title>Bench4KE</title>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
      <link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

<div class="container py-4">
  <h1 class="mb-4">Bench4KE Validator</h1>
  <div class="d-flex gap-3 mb-4">
  <a class="btn btn-outline-dark"
     href="https://github.com/fossr-project/ontogenia-cini/tree/main"
     target="_blank" rel="noopener">
    <i class="bi bi-github me-2"></i>GitHub
  </a>

  <a class="btn btn-outline-primary"
     href="https://docs.google.com/forms/d/e/1FAIpQLSfpYHGzA2r0wKCq0xEVIkPBKKol6umiKn1URAc17f709DKMKg/viewform?usp=dialog"
     target="_blank" rel="noopener">
    Tell us what you think
  </a>
</div>


  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <div class="alert alert-warning mt-3">{{ messages[0] }}</div>
    {% endif %}
  {% endwith %}

  <ul class="nav nav-tabs" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link {% if active_tab == 'cq' %}active{% endif %}" id="cq-tab" data-bs-toggle="tab" data-bs-target="#cq-panel" type="button" role="tab">CQ Validation</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link {% if active_tab == 'generate' %}active{% endif %}" id="generate-tab" data-bs-toggle="tab" data-bs-target="#generate-panel" type="button" role="tab">Ontology Generate</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link {% if active_tab == 'ontology' %}active{% endif %}" id="ontology-tab" data-bs-toggle="tab" data-bs-target="#ontology-panel" type="button" role="tab">Ontology Benchmark</button>
    </li>
  </ul>

  <div class="tab-content border border-top-0 p-3 bg-white">
    <div class="tab-pane fade {% if active_tab == 'cq' %}show active{% endif %}" id="cq-panel" role="tabpanel">
      <form class="mt-2" method="post" action="{{ url_for('validate') }}">
        <div class="mb-3">
          <label for="external_url" class="form-label">CQ Generator API URL:</label>
          <input class="form-control" type="url" id="external_url" name="external_url" placeholder="https://.../newapi/" required>
        </div>

        <button class="btn btn-primary" type="submit">Validate with Default Dataset</button>
      </form>

      {% if result %}
        <hr>
        <h2 class="h4">Validation Result</h2>
        <pre class="border p-3 bg-light">{{ result | tojson(indent=2) }}</pre>

        {% for r in result.validation_results %}
          {% if r.Cosine_Heatmap %}
            <h5 class="mt-4">Cosine heat-map</h5>
            <img class="img-fluid" src="{{ r.Cosine_Heatmap }}">
          {% endif %}
          {% if r.Jaccard_Heatmap %}
            <h5 class="mt-4">Jaccard heat-map</h5>
            <img class="img-fluid" src="{{ r.Jaccard_Heatmap }}">
          {% endif %}
        {% endfor %}
      {% endif %}
    </div>

    <div class="tab-pane fade {% if active_tab == 'generate' %}show active{% endif %}" id="generate-panel" role="tabpanel">
      <form class="mt-2" method="post" action="{{ url_for('generate_ontology') }}">
        <div class="mb-3">
          <label for="generator_url" class="form-label">Ontology Adapter URL:</label>
          <input class="form-control" type="url" id="generator_url" name="generator_url" placeholder="http://127.0.0.1:8020/generate_ontology" required>
          <div class="form-text">This calls the generator directly (no benchmark/evaluation).</div>
        </div>

        <div class="row g-3">
          <div class="col-md-4">
            <label for="gen_system" class="form-label">System:</label>
            <select class="form-select" id="gen_system" name="gen_system" required>
              <option value="ontogenia">ontogenia</option>
              <option value="ontogenia-mp">ontogenia-mp</option>
              <option value="domain-ontogen">domain-ontogen</option>
              <option value="neon-gpt">neon-gpt</option>
              <option value="neon-gpt-llms4life">neon-gpt-llms4life</option>
            </select>
          </div>
          <div class="col-md-4">
            <label for="dataset_id" class="form-label">dataset_id (optional):</label>
            <input class="form-control" type="text" id="dataset_id" name="dataset_id" placeholder="my-item-1">
          </div>
          <div class="col-md-4">
            <label for="scenario_id" class="form-label">scenario_id (optional):</label>
            <input class="form-control" type="text" id="scenario_id" name="scenario_id" placeholder="scenario-1">
          </div>
        </div>

        <div class="mb-3 mt-3">
          <label for="scenario" class="form-label">scenario (optional):</label>
          <textarea class="form-control" id="scenario" name="scenario" rows="3" placeholder="Domain description / story..."></textarea>
        </div>

        <div class="mb-3">
          <label for="competency_questions" class="form-label">competency_questions (one per line):</label>
          <textarea class="form-control" id="competency_questions" name="competency_questions" rows="4" required placeholder="CQ 1&#10;CQ 2&#10;CQ 3"></textarea>
        </div>

        <div class="mb-3">
          <label for="user_stories" class="form-label">user_stories (one per line, optional):</label>
          <textarea class="form-control" id="user_stories" name="user_stories" rows="3" placeholder="User story 1&#10;User story 2"></textarea>
        </div>

        <div class="row g-3">
          <div class="col-md-3">
            <label for="output_format" class="form-label">constraints.output_format:</label>
            <select class="form-select" id="output_format" name="output_format">
              <option value="ttl" selected>ttl</option>
              <option value="rdfxml">rdfxml</option>
              <option value="owl">owl</option>
              <option value="jsonld">jsonld</option>
            </select>
          </div>
          <div class="col-md-3">
            <label for="iri_base" class="form-label">constraints.iri_base:</label>
            <input class="form-control" type="text" id="iri_base" name="iri_base" placeholder="http://example.org/onto#">
          </div>
          <div class="col-md-3">
            <label for="naming_policy" class="form-label">constraints.naming_policy:</label>
            <input class="form-control" type="text" id="naming_policy" name="naming_policy" placeholder="camelCase">
          </div>
          <div class="col-md-3">
            <label for="language" class="form-label">constraints.language:</label>
            <input class="form-control" type="text" id="language" name="language" placeholder="en">
          </div>
        </div>

        <div class="mb-3 mt-3">
          <label for="metadata" class="form-label">metadata (JSON, optional):</label>
          <textarea class="form-control" id="metadata" name="metadata" rows="4" placeholder='{"model":"gpt-4o-mini"}'></textarea>
        </div>

        <div class="form-check mb-3">
          <input class="form-check-input" type="checkbox" value="1" id="raw_output" name="raw_output">
          <label class="form-check-label" for="raw_output">raw_output (skip Turtle normalization)</label>
        </div>

        <button class="btn btn-primary" type="submit">Generate Ontology</button>
      </form>

      {% if generate_result %}
        <hr>
        <h2 class="h4">Generation Result</h2>
        <pre class="border p-3 bg-light">{{ generate_result | tojson(indent=2) }}</pre>
      {% endif %}
    </div>

    <div class="tab-pane fade {% if active_tab == 'ontology' %}show active{% endif %}" id="ontology-panel" role="tabpanel">
      <form class="mt-2" method="post" action="{{ url_for('run_ontology') }}">
        <div class="mb-3">
          <label for="ontology_url" class="form-label">Ontology Generator API URL:</label>
          <input class="form-control" type="url" id="ontology_url" name="ontology_url" placeholder="https://.../generate_ontology" required>
        </div>
        <div class="mb-3">
          <label for="dataset_path" class="form-label">Dataset path (optional):</label>
          <input class="form-control" type="text" id="dataset_path" name="dataset_path" placeholder="datasets/ontology_generation/normalized/ontogenia.jsonl">
          <div class="form-text">If empty, the API uses <code>ONTOLOGY_DATASET_DIR</code>. You can point to a single <code>.jsonl</code> file or a directory of <code>.jsonl</code> files.</div>
        </div>
        <div class="mb-3">
          <label for="items_json" class="form-label">Inline items (JSON, optional):</label>
          <textarea class="form-control" id="items_json" name="items_json" rows="5" placeholder='[{"scenario":"...","competency_questions":["..."]}]'></textarea>
          <div class="form-text">If provided, the benchmark runs exactly these items (and ignores dataset loading).</div>
        </div>
        <div class="mb-3">
          <label for="system" class="form-label">System filter (optional):</label>
          <input class="form-control" type="text" id="system" name="system" placeholder="ontogenia | ontogenia-mp | domain-ontogen | neon-gpt | neon-gpt-llms4life | all">
        </div>
        <div class="mb-3">
          <label for="domain_ontogen_mode" class="form-label">Domain-OntoGen mode (optional):</label>
          <select class="form-select" id="domain_ontogen_mode" name="domain_ontogen_mode">
            <option value="" selected>(default: per_item)</option>
            <option value="per_item">per_item (one ontology per dataset item)</option>
            <option value="per_cq">per_cq (one ontology per CQ; paper-faithful)</option>
          </select>
          <div class="form-text">Applies only when the evaluated system is <code>domain-ontogen</code>.</div>
        </div>
        <div class="mb-3">
          <label for="evaluation_mode" class="form-label">Evaluation mode:</label>
          <input class="form-control" type="text" id="evaluation_mode" name="evaluation_mode" value="all">
        </div>
        <div class="mb-3">
          <label for="max_items" class="form-label">Max items (optional):</label>
          <input class="form-control" type="number" id="max_items" name="max_items" min="0" step="1" placeholder="0 = all">
        </div>
        <div class="mb-3">
          <label for="model" class="form-label">Generator model override (optional):</label>
          <input class="form-control" type="text" id="model" name="model" placeholder="gpt-4o-mini">
          <div class="form-text">Passed to the external generator as <code>metadata.model</code> (run-level default).</div>
        </div>
        <div class="mb-3">
          <label for="llm_eval_model" class="form-label">LLM eval model (optional):</label>
          <input class="form-control" type="text" id="llm_eval_model" name="llm_eval_model" placeholder="gpt-4o">
        </div>

        <button class="btn btn-primary" type="submit">Run Ontology Benchmark</button>
      </form>

      {% if ontology_result %}
        <hr>
        <h2 class="h4">Ontology Benchmark Result</h2>
        <pre class="border p-3 bg-light">{{ ontology_result | tojson(indent=2) }}</pre>
      {% endif %}
    </div>
  </div>
</div>
"""

# ---- routes ----
@app.route("/")
def index():
    return render_template_string(
        TEMPLATE,
        result=None,
        generate_result=None,
        ontology_result=None,
        active_tab="cq",
    )

@app.post("/validate")
def validate():
    external_url = request.form.get("external_url", "").strip()
    if not external_url:
        flash("Please provide the CQ generator API URL.")
        return redirect(url_for("index"))

    data = {
        "use_default_dataset": "true",
        "external_service_url": external_url,
        "validation_mode": "all",
        "model": DEFAULT_OPENAI_MODEL,
        "save_results": "true"
    }
    api_key = request.form.get("api_key", os.getenv("EXTERNAL_CQ_API_KEY", "")).strip()
    if api_key:
        data["api_key"] = api_key

    try:
        connect_tm = 10
        read_tm = None if API_TIMEOUT == 0 else API_TIMEOUT
        resp = requests.post(
            f"{API_BASE}/validate/",
            data=data,
            timeout=(connect_tm, read_tm)
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as exc:
        flash(f"Error contacting Validator API: {exc}")
        return redirect(url_for("index"))

    return render_template_string(
        TEMPLATE,
        result=result,
        generate_result=None,
        ontology_result=None,
        active_tab="cq",
    )


@app.post("/generate")
def generate_ontology():
    generator_url = request.form.get("generator_url", "").strip()
    if not generator_url:
        flash("Please provide the ontology adapter URL.")
        return redirect(url_for("index"))

    system = (request.form.get("gen_system") or "").strip()
    if not system:
        flash("Please select a system.")
        return redirect(url_for("index"))

    dataset_id = (request.form.get("dataset_id") or "").strip() or None
    scenario_id = (request.form.get("scenario_id") or "").strip() or None
    scenario = (request.form.get("scenario") or "").strip() or None

    cqs_raw = request.form.get("competency_questions") or ""
    competency_questions = [line.strip() for line in cqs_raw.splitlines() if line.strip()]
    if not competency_questions:
        flash("Please provide at least one competency question.")
        return redirect(url_for("index"))

    stories_raw = request.form.get("user_stories") or ""
    user_stories = [line.strip() for line in stories_raw.splitlines() if line.strip()] or None

    constraints = {
        "output_format": (request.form.get("output_format") or "ttl").strip() or "ttl",
    }
    iri_base = (request.form.get("iri_base") or "").strip()
    naming_policy = (request.form.get("naming_policy") or "").strip()
    language = (request.form.get("language") or "").strip()
    if iri_base:
        constraints["iri_base"] = iri_base
    if naming_policy:
        constraints["naming_policy"] = naming_policy
    if language:
        constraints["language"] = language

    metadata_text = (request.form.get("metadata") or "").strip()
    metadata = None
    if metadata_text:
        try:
            metadata = json.loads(metadata_text)
        except Exception as exc:
            flash(f"metadata must be valid JSON: {exc}")
            return redirect(url_for("index"))

    raw_output = bool((request.form.get("raw_output") or "").strip())

    payload = {
        "system": system,
        "dataset_id": dataset_id,
        "scenario_id": scenario_id,
        "scenario": scenario,
        "competency_questions": competency_questions,
        "user_stories": user_stories,
        "constraints": constraints,
        "metadata": metadata or {},
        "raw_output": raw_output,
    }
    # Remove null fields to keep the request clean.
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        connect_tm = 10
        read_tm = None if API_TIMEOUT == 0 else API_TIMEOUT
        resp = requests.post(generator_url, json=payload, timeout=(connect_tm, read_tm))
        resp.raise_for_status()
        generate_result = resp.json()
    except Exception as exc:
        flash(f"Error contacting ontology adapter: {exc}")
        return redirect(url_for("index"))

    return render_template_string(
        TEMPLATE,
        result=None,
        generate_result=generate_result,
        ontology_result=None,
        active_tab="generate",
    )


@app.post("/ontology")
def run_ontology():
    external_url = request.form.get("ontology_url", "").strip()
    if not external_url:
        flash("Please provide the ontology generator API URL.")
        return redirect(url_for("index"))

    payload = {
        "use_default_dataset": True,
        "evaluation_mode": (request.form.get("evaluation_mode") or "all").strip(),
        "external_service_url": external_url,
        "save_results": True,
    }
    items_json = (request.form.get("items_json") or "").strip()
    if items_json:
        try:
            payload["items"] = json.loads(items_json)
            payload["use_default_dataset"] = False
        except Exception as exc:
            flash(f"Inline items must be valid JSON: {exc}")
            return redirect(url_for("index"))
    else:
        dataset_path = (request.form.get("dataset_path") or "").strip()
        if dataset_path:
            payload["dataset_path"] = dataset_path
    system = (request.form.get("system") or "").strip()
    if system:
        payload["system"] = system
    domain_ontogen_mode = (request.form.get("domain_ontogen_mode") or "").strip()
    if domain_ontogen_mode:
        payload["domain_ontogen_mode"] = domain_ontogen_mode
    max_items = (request.form.get("max_items") or "").strip()
    if max_items:
        try:
            payload["max_items"] = int(max_items)
        except ValueError:
            flash("Max items must be an integer.")
            return redirect(url_for("index"))
    model = (request.form.get("model") or "").strip()
    if model:
        payload["model"] = model
    llm_eval_model = (request.form.get("llm_eval_model") or "").strip()
    if llm_eval_model:
        payload["llm_eval_model"] = llm_eval_model

    try:
        connect_tm = 10
        read_tm = None if API_TIMEOUT == 0 else API_TIMEOUT
        resp = requests.post(
            f"{API_BASE}/ontology/run",
            json=payload,
            timeout=(connect_tm, read_tm)
        )
        resp.raise_for_status()
        ontology_result = resp.json()
        # Enrich the UI response by loading run metadata and metric JSONs from disk,
        # so the page shows actual statistics (not only file paths).
        run_dir = ontology_result.get("run_dir") if isinstance(ontology_result, dict) else None
        if run_dir:
            run_meta = _load_json_file(os.path.join(run_dir, "run_metadata.json"))
            if run_meta is not None:
                ontology_result["run_metadata"] = run_meta
        results = ontology_result.get("results") if isinstance(ontology_result, dict) else None
        if isinstance(results, list):
            for r in results:
                if not isinstance(r, dict):
                    continue
                ontometrics_path = r.get("ontometrics_file")
                oops_path = r.get("oops_file")
                llm_path = r.get("llm_eval_file")
                ontometrics_obj = _load_json_file(ontometrics_path) if ontometrics_path else None
                if ontometrics_obj is not None:
                    r["ontometrics"] = ontometrics_obj
                oops_obj = _load_json_file(oops_path) if oops_path else None
                if oops_obj is not None:
                    r["oops"] = _summarize_oops(oops_obj)
                llm_obj = _load_json_file(llm_path) if llm_path else None
                if llm_obj is not None:
                    r["llm_eval"] = _summarize_llm_eval(llm_obj)
    except Exception as exc:
        flash(f"Error contacting Ontology API: {exc}")
        return redirect(url_for("index"))

    return render_template_string(
        TEMPLATE,
        result=None,
        generate_result=None,
        ontology_result=ontology_result,
        active_tab="ontology",
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
