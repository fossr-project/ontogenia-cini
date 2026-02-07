import json
from pathlib import Path
from typing import List

from app.models_ontology import OntologyGenerationItem


def load_ontology_items(dataset_path: str) -> List[OntologyGenerationItem]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    files = [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files found in: {path}")

    items: List[OntologyGenerationItem] = []
    for file_path in files:
        items.extend(_load_jsonl_file(file_path))
    return items


def _load_jsonl_file(path: Path) -> List[OntologyGenerationItem]:
    items: List[OntologyGenerationItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            item_payload = _normalize_payload(payload)
            # Preserve dataset provenance so the benchmark can aggregate results
            # by dataset/system/model (as required by the project spec).
            metadata = dict(item_payload.get("metadata") or {})
            metadata.setdefault("dataset_name", path.name)
            item_payload["metadata"] = metadata
            try:
                items.append(OntologyGenerationItem(**item_payload))
            except Exception as exc:
                raise ValueError(
                    f"Invalid ontology item on line {line_no} in {path}: {exc}"
                ) from exc
    return items


def _normalize_payload(payload: dict) -> dict:
    item = dict(payload)
    if "dataset_id" not in item and "id" in item:
        item["dataset_id"] = str(item["id"]).strip()
    item.pop("id", None)

    prompt_template = item.pop("prompt_template", None)
    if prompt_template:
        metadata = dict(item.get("metadata") or {})
        metadata.setdefault("prompt_template", prompt_template)
        item["metadata"] = metadata
    return item
