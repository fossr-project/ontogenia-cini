"""Consistency-evaluation router.

Invocation mirrors the `/validate/` service with multipart POST with
form parameters (and an optional uploaded file), not a JSON body:

    curl -X POST http://127.0.0.1:8000/consistency/ \
         -F "config_file=@conf/consistency.yaml" \
         -F "override=evaluation.n_runs=3" \
         -F "override=dataset.max_rows=5"
    # -> {"job_id": "...", "status": "queued", "poll_url": "/consistency/<id>"}

    curl http://127.0.0.1:8000/consistency/<job_id>     # poll status/result

The uploaded `config_file` is the reproducible **baseline**; repeated `override`
form fields tweak single parameters via `key.path=value`. If no configuration
file is provided the built-in defaults are used.

Progress on jobs can be requested via the GET endpoint.
"""

import logging
import uuid
from collections import OrderedDict
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.models import (
    ConsistencyConfig,
    ConsistencyJobResponse,
    ConsistencyStatusResponse,
)
from app.services.consistency_config import load_config_file, resolve_config
from app.services.consistency_evaluator import ConsistencyEvaluator

logger = logging.getLogger(__name__)
router = APIRouter()

_jobs: "OrderedDict[str, dict]" = OrderedDict()
_MAX_JOBS = 100


def _consistency_task(job_id: str, config: ConsistencyConfig) -> None:
    """Runs in threadpool. Mutates the job dict in place."""
    def progress(event: dict) -> None:
        if job_id in _jobs:
            _jobs[job_id]["progress"] = event

    if job_id not in _jobs:
        return
    try:
        _jobs[job_id]["status"] = "running"
        result = ConsistencyEvaluator(config, progress_callback=progress).run_all()
        if job_id in _jobs:
            _jobs[job_id].update(status="complete", result=result)
    except Exception as e:
        if job_id in _jobs:
            _jobs[job_id].update(status="failed", error=str(e))
        logger.error("Consistency job %s failed: %s", job_id, e)


@router.post("/")
async def run_consistency(
    background_tasks: BackgroundTasks,
    config_file: UploadFile = File(None),       # optional baseline config (YAML/JSON)
    override: Optional[List[str]] = Form(None), # repeated key.path=value single-param overrides
):
    """Submit a consistency-evaluation job built from an optional baseline config
    file plus single-parameter overrides. Returns a job id to poll."""
    file_dict = None
    if config_file is not None:
        try:
            content = await config_file.read()
            file_dict = load_config_file(content, config_file.filename or "")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config file: {e}")

    try:
        config = resolve_config(file_dict=file_dict, overrides=override or [])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "progress": None, "result": None, "error": None}
    while len(_jobs) > _MAX_JOBS:
        _jobs.popitem(last=False)
    background_tasks.add_task(_consistency_task, job_id, config)

    return ConsistencyJobResponse(
        job_id=job_id, status="queued", poll_url=f"/consistency/{job_id}",
    )


@router.get("/{job_id}")
async def get_consistency_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return ConsistencyStatusResponse(job_id=job_id, **_jobs[job_id])
