import json
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ontology_run_smoke(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    ttl = (
        "@prefix : <http://example.org/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":Test a owl:Class .\n"
    )

    def stub_external_service(payload, external_url, timeout):
        return {
            "ontology": {"format": "ttl", "content": ttl},
            "metadata": {
                "system_name": "stub",
                "model": "stub",
                "duration_ms": 1,
                "timestamp": "2026-01-01T00:00:00Z",
            },
        }

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "competency_questions": ["What is X?"],
            }
        ],
        "evaluation_mode": "ontometrics",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Processing complete"
    assert data["run_dir"]

    summary_path = os.path.join(data["run_dir"], "summary.json")
    assert os.path.isfile(summary_path)

    result = data["results"][0]
    assert result["error"] is None
    assert result["ontology_file"]
    assert os.path.isfile(result["ontology_file"])
    assert result["ontometrics_file"]
    assert os.path.isfile(result["ontometrics_file"])
    assert result["oops_file"] is None
    assert result["llm_eval_file"] is None


def test_ontology_run_all_metrics(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    ttl = (
        "@prefix : <http://example.org/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":Test a owl:Class .\n"
    )

    def stub_external_service(payload, external_url, timeout):
        return {
            "ontology": {"format": "ttl", "content": ttl},
            "metadata": {
                "system_name": "stub",
                "model": "stub",
                "duration_ms": 1,
                "timestamp": "2026-01-01T00:00:00Z",
            },
        }

    def stub_oops_scan(content, api_url, timeout, mode):
        return {"pitfalls": [{"code": "P10", "severity": "minor"}], "count": 1}

    def stub_llm_eval(**kwargs):
        results = [
            {"label": "yes", "sparql": "SELECT * WHERE { ?s ?p ?o }"},
            {"label": "no", "sparql": ""},
        ]
        summary = {"yes": 1, "no": 1, "total": 2, "yes_ratio": 0.5}
        return results, summary

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "run_oops_scan", stub_oops_scan)
    monkeypatch.setattr(ontology_benchmark, "evaluate_ontology_with_llm", stub_llm_eval)
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "competency_questions": ["What is X?"],
            }
        ],
        "evaluation_mode": "all",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    result = data["results"][0]
    assert result["error"] is None
    assert result["ontology_file"]
    assert result["ontometrics_file"]
    assert result["oops_file"]
    assert result["llm_eval_file"]
    assert os.path.isfile(result["ontology_file"])
    assert os.path.isfile(result["ontometrics_file"])
    assert os.path.isfile(result["oops_file"])
    assert os.path.isfile(result["llm_eval_file"])

    summary_path = data["results_saved_to"]
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["llm_eval_aggregate"]["yes"] == 1
    assert summary["llm_eval_aggregate"]["total"] == 2

    sparql_dir = os.path.join(data["run_dir"], "metrics", "sparql_queries")
    sparql_files = os.listdir(sparql_dir)
    assert sparql_files
    sparql_path = os.path.join(sparql_dir, sparql_files[0])
    with open(sparql_path, "r", encoding="utf-8") as handle:
        sparql_text = handle.read()
    assert "SELECT" in sparql_text


def test_ontology_run_llm_mode_creates_sparql(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    ttl = (
        "@prefix : <http://example.org/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":Test a owl:Class .\n"
    )

    def stub_external_service(payload, external_url, timeout):
        return {"ontology": {"format": "ttl", "content": ttl}, "metadata": {}}

    def stub_llm_eval(**kwargs):
        results = [{"label": "yes", "sparql": "SELECT * WHERE { ?s ?p ?o }"}]
        summary = {"yes": 1, "no": 0, "total": 1, "yes_ratio": 1.0}
        return results, summary

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Unexpected metric call")

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "evaluate_ontology_with_llm", stub_llm_eval)
    monkeypatch.setattr(ontology_benchmark, "compute_ontometrics", unexpected_call)
    monkeypatch.setattr(ontology_benchmark, "run_oops_scan", unexpected_call)
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "competency_questions": ["What is X?"],
            }
        ],
        "evaluation_mode": "llm",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    result = data["results"][0]
    assert result["ontometrics_file"] is None
    assert result["oops_file"] is None
    assert result["llm_eval_file"]

    sparql_dir = os.path.join(data["run_dir"], "metrics", "sparql_queries")
    sparql_files = os.listdir(sparql_dir)
    assert sparql_files
    sparql_path = os.path.join(sparql_dir, sparql_files[0])
    with open(sparql_path, "r", encoding="utf-8") as handle:
        sparql_text = handle.read()
    assert "SELECT" in sparql_text


def test_ontology_run_evaluation_mode_invalid_ignored(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    ttl = (
        "@prefix : <http://example.org/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":Test a owl:Class .\n"
    )

    def stub_external_service(payload, external_url, timeout):
        return {
            "ontology": {"format": "ttl", "content": ttl},
            "metadata": {"system_name": "stub"},
        }

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Unexpected metric call")

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "run_oops_scan", unexpected_call)
    monkeypatch.setattr(ontology_benchmark, "evaluate_ontology_with_llm", unexpected_call)
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "competency_questions": ["What is X?"],
            }
        ],
        "evaluation_mode": "ontometrics,invalid",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    result = data["results"][0]
    assert result["ontometrics_file"]
    assert result["oops_file"] is None
    assert result["llm_eval_file"] is None


def test_ontology_run_save_results_false(monkeypatch):
    import app.routers.ontology_benchmark as ontology_benchmark

    ttl = (
        "@prefix : <http://example.org/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":Test a owl:Class .\n"
    )

    def stub_external_service(payload, external_url, timeout):
        return {"ontology": {"format": "ttl", "content": ttl}, "metadata": {}}

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "competency_questions": ["What is X?"],
            }
        ],
        "evaluation_mode": "ontometrics",
        "save_results": False,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["run_dir"] is None
    assert data["results_saved_to"] == "Not saved"
    result = data["results"][0]
    assert result["ontology_file"] is None
    assert result["ontometrics_file"] is None
    assert result["oops_file"] is None
    assert result["llm_eval_file"] is None


def test_ontology_run_system_filter_and_max_items(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    ttl = (
        "@prefix : <http://example.org/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":Test a owl:Class .\n"
    )

    def stub_external_service(payload, external_url, timeout):
        return {"ontology": {"format": "ttl", "content": ttl}, "metadata": {}}

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "ontogenia",
                "competency_questions": ["Q1"],
            },
            {
                "dataset_id": "item-2",
                "system": "domain-ontogen",
                "competency_questions": ["Q2"],
            },
            {
                "dataset_id": "item-3",
                "competency_questions": ["Q3"],
            },
        ],
        "system": "ontogenia",
        "max_items": 1,
        "evaluation_mode": "ontometrics",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["system"] == "ontogenia"


def test_ontology_run_ontometrics_parse_error(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    def stub_external_service(payload, external_url, timeout):
        return {"ontology": {"format": "ttl", "content": "not ttl"}, "metadata": {}}

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "competency_questions": ["What is X?"],
            }
        ],
        "evaluation_mode": "ontometrics",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    result = data["results"][0]
    with open(result["ontometrics_file"], "r", encoding="utf-8") as handle:
        ontometrics = json.load(handle)
    assert "error" in ontometrics


def test_ontology_run_oops_skipped_when_no_url(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    ttl = (
        "@prefix : <http://example.org/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":Test a owl:Class .\n"
    )

    def stub_external_service(payload, external_url, timeout):
        return {"ontology": {"format": "ttl", "content": ttl}, "metadata": {}}

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(ontology_benchmark, "OOPS_API_URL", "")

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "competency_questions": ["What is X?"],
            }
        ],
        "evaluation_mode": "oops",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    result = data["results"][0]
    with open(result["oops_file"], "r", encoding="utf-8") as handle:
        oops_data = json.load(handle)
    assert oops_data.get("skipped") is True


def test_ontology_run_domain_ontogen_per_cq_expands_items(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    ttl = (
        "@prefix : <http://example.org/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":Test a owl:Class .\n"
    )

    calls = {"n": 0, "payloads": []}

    def stub_external_service(payload, external_url, timeout):
        calls["n"] += 1
        calls["payloads"].append(payload)
        return {"ontology": {"format": "ttl", "content": ttl}, "metadata": {"model": "stub"}}

    def stub_llm_eval(**kwargs):
        results = [{"label": "yes", "sparql": "SELECT * WHERE { ?s ?p ?o }"}]
        summary = {"yes": 1, "no": 0, "total": 1, "yes_ratio": 1.0}
        return results, summary

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "evaluate_ontology_with_llm", stub_llm_eval)
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "scenario": "Story",
                "competency_questions": ["CQ1", "CQ2", "CQ3"],
            }
        ],
        "domain_ontogen_mode": "per_cq",
        "evaluation_mode": "llm",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert calls["n"] == 3
    assert len(data["results"]) == 3
    for p in calls["payloads"]:
        assert p["system"] == "domain-ontogen"
        assert len(p["competency_questions"]) == 1

    with open(data["results_saved_to"], "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["llm_eval_aggregate"]["total"] == 3


def test_ontology_run_external_error(monkeypatch, tmp_path):
    import app.routers.ontology_benchmark as ontology_benchmark

    def stub_external_service(payload, external_url, timeout):
        raise Exception("boom")

    monkeypatch.setattr(
        ontology_benchmark, "call_external_ontology_service", stub_external_service
    )
    monkeypatch.setattr(ontology_benchmark, "ONTOLOGY_RUNS_DIR", str(tmp_path))

    payload = {
        "items": [
            {
                "dataset_id": "item-1",
                "system": "domain-ontogen",
                "competency_questions": ["What is X?"],
            }
        ],
        "evaluation_mode": "ontometrics",
        "save_results": True,
    }

    response = client.post("/ontology/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    result = data["results"][0]
    assert result["error"]
    assert result["ontology_file"] is None
    assert os.path.isfile(data["results_saved_to"])
