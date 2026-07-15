#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from app.services.consistency_config import load_config_file, resolve_config

_DEFAULT_CONFIG = os.path.join(_HERE, "conf", "consistency.yaml")


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="ConsistencyEvaluator CLI")
    parser.add_argument("--submit", action="store_true",
                        help="POST the config to the server and poll to completion")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000",
                        help="FastAPI base URL (used with --submit)")
    parser.add_argument("--config", default=None,
                        help="baseline config file (YAML/JSON); "
                             "defaults to conf/consistency.yaml if present")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra-style key.path=value single-parameter overrides")
    return parser.parse_args(argv)


def _resolve_config_path(path):
    if path:
        return path
    return _DEFAULT_CONFIG if os.path.exists(_DEFAULT_CONFIG) else None


def _load_file_dict(path):
    if not path:
        return None
    with open(path, "rb") as f:
        return load_config_file(f.read(), path)


def _submit_and_poll(overrides, config, config_path, server_url):
    import requests  # local import — only needed for --submit

    base = server_url.rstrip("/")
    files = {}
    if config_path and os.path.exists(config_path):
        files["config_file"] = (os.path.basename(config_path), open(config_path, "rb"))
    data = [("override", o) for o in overrides]
    resp = requests.post(f"{base}/consistency/", files=files or None, data=data or None)
    resp.raise_for_status()
    job = resp.json()
    job_id, poll_url = job["job_id"], f"{base}{job['poll_url']}"
    print(f"{datetime.now().strftime('%H:%M:%S')} — [runner] submitted job {job_id}; polling every {config.poll_interval}s …")

    while True:
        status_data = requests.get(poll_url).json()
        status = status_data["status"]
        if status == "complete":
            os.makedirs(config.output_dir, exist_ok=True)
            out = os.path.join(config.output_dir, f"result_{job_id}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(status_data["result"], f, indent=2, ensure_ascii=False)
            print(f"{datetime.now().strftime('%H:%M:%S')} — [runner] complete → {out}")
            return status_data["result"]
        if status == "failed":
            raise RuntimeError(f"Consistency job failed: {status_data.get('error')}")
        prog = status_data.get("progress") or {}
        if prog:
            print(
                f"{datetime.now().strftime('%H:%M:%S')} — [runner] [{prog.get('test')}] run {prog.get('run')}/"
                f"{prog.get('total_runs')} — task {prog.get('completed_tasks')}/{prog.get('total_tasks')} — "
                f"{prog.get('phase')} — workers {prog.get('active_workers')}/{prog.get('max_workers')}"
            )
        time.sleep(config.poll_interval)


def main(argv=None):
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    overrides = [o for o in args.overrides if "=" in o]
    config_path = _resolve_config_path(args.config)
    file_dict = _load_file_dict(config_path)

    config = resolve_config(file_dict=file_dict, overrides=overrides)

    if not args.submit:
        # dry run: resolved config only
        print(config.json(indent=2))
        return config
    return _submit_and_poll(overrides, config, config_path, args.server_url)


if __name__ == "__main__":
    main()
