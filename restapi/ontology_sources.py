import logging
import os
from typing import List, Tuple
from urllib.parse import urlparse

import requests

ONTOLOGY_EXTENSIONS = (".owl", ".ttl", ".rdf", ".xml", ".n3", ".nt")
REQUEST_TIMEOUT = 45
logger = logging.getLogger(__name__)
_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _http_get(url: str) -> bytes:
    headers = {}
    if _GITHUB_TOKEN and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {_GITHUB_TOKEN}"
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content



def _parse_github_tree(url: str) -> Tuple[str, str, str, str]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if "tree" not in parts:
        raise ValueError("GitHub tree URL must contain /tree/<branch>/path segments.")
    tree_idx = parts.index("tree")
    owner = parts[0]
    repo = parts[1]
    branch = parts[tree_idx + 1]
    rel_path = "/".join(parts[tree_idx + 2:])
    return owner, repo, branch, rel_path


def _github_api_headers() -> dict:
    if not _GITHUB_TOKEN:
        return {}
    return {"Authorization": f"Bearer {_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}


def _fetch_github_directory(owner: str, repo: str, branch: str, path: str) -> List[Tuple[str, bytes]]:
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    resp = requests.get(api_url, headers=_github_api_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    files: List[Tuple[str, bytes]] = []
    for entry in data:
        if entry.get("type") == "dir":
            files.extend(_fetch_github_directory(owner, repo, branch, entry["path"]))
        elif entry.get("type") == "file":
            name = entry.get("name", "")
            if not name.lower().endswith(ONTOLOGY_EXTENSIONS):
                continue
            download_url = entry.get("download_url")
            if not download_url:
                continue
            logger.info("Downloading %s from GitHub repo %s/%s", entry.get("path"), owner, repo)
            content = _http_get(download_url)
            files.append((name, content))
    return files


def fetch_ontology_resources(link: str) -> List[Tuple[str, bytes]]:
    """Return a list of (name, bytes) ontology files referenced by the link."""
    if not link:
        return []
    link = link.strip()
    if link.startswith("http") and any(link.lower().endswith(ext) for ext in ONTOLOGY_EXTENSIONS):
        name = os.path.basename(urlparse(link).path) or "ontology.owl"
        return [(name, _http_get(link))]
    if link.startswith("file://") or link.startswith("/"):
        path = link[7:] if link.startswith("file://") else link
        if not os.path.exists(path):
            logger.warning("Local ontology file %s not found", path)
            return []
        with open(path, "rb") as f:
            return [(os.path.basename(path), f.read())]
    if link.startswith("https://github.com/") and "/tree/" in link:
        owner, repo, branch, rel_path = _parse_github_tree(link)
        return _fetch_github_directory(owner, repo, branch, rel_path)
    logger.warning("Unsupported ontology link format: %s", link)
    return []
