from flask import Flask, render_template_string, request, redirect, url_for, flash
import requests, os, json
from dotenv import load_dotenv
load_dotenv()
API_TIMEOUT = int(os.getenv("CQ_API_TIMEOUT", "3600"))

API_BASE = os.environ.get("CQ_API_URL", "http://127.0.0.1:8000")
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "changeme")

TEMPLATE = """
<!doctype html>
<title>Bench4KE</title>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
      <link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">

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

  <form class="mt-4" method="post" action="{{ url_for('validate') }}">
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
"""

# ---- routes ----
@app.route("/")
def index():
    return render_template_string(TEMPLATE, result=None)

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
        "output_folder": "heatmaps",
        "model": "gpt-4",
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

    return render_template_string(TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
