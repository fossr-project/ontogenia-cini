import hashlib
import os
import pickle
from urllib.parse import urlparse

import numpy as np
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
DOWNLOADS_DIR = os.path.join(CACHE_DIR, "downloads")
EMBED_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings.pkl")
BERTSCORE_CACHE_PATH = os.path.join(CACHE_DIR, "bertscore.pkl")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _load_pickle(path: dict) -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_pickle(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


class EmbeddingCache:
    def __init__(self, sbert_model, model_name: str = "all-MiniLM-L6-v2"):
        self.model = sbert_model
        self.model_name = model_name
        self._store = _load_pickle(EMBED_CACHE_PATH)
        self._dirty = False

    def encode(self, texts: list) -> np.ndarray:
        keys = [_hash(self.model_name, t) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self._store]
        if missing:
            new_embs = self.model.encode(missing, convert_to_numpy=True)
            for t, emb in zip(missing, new_embs):
                self._store[_hash(self.model_name, t)] = emb
            self._dirty = True
        return np.stack([self._store[k] for k in keys])

    def flush(self) -> None:
        if self._dirty:
            _save_pickle(self._store, EMBED_CACHE_PATH)
            self._dirty = False


class BertscoreCache:
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        self.model_type = model_type
        self._store = _load_pickle(BERTSCORE_CACHE_PATH)
        self._dirty = False

    def score(self, cands: list, refs: list) -> np.ndarray:
        import bert_score

        keys = [_hash(self.model_type, c, r) for c, r in zip(cands, refs)]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._store]

        if missing_idx:
            m_cands = [cands[i] for i in missing_idx]
            m_refs = [refs[i] for i in missing_idx]
            _, _, F1 = bert_score.score(
                m_cands, m_refs, lang="en", model_type=self.model_type,
                verbose=False, batch_size=64,
            )
            for i, f1 in zip(missing_idx, F1.numpy()):
                self._store[keys[i]] = float(f1)
            self._dirty = True

        return np.array([self._store[k] for k in keys])

    def flush(self) -> None:
        if self._dirty:
            _save_pickle(self._store, BERTSCORE_CACHE_PATH)
            self._dirty = False


def cached_download(url: str, timeout: int = 20) -> str:
    """Downloads url once and caches it under eval_results/.cache/downloads/.
    Returns the local file path. Safe to call repeatedly across runs."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    ext = os.path.splitext(urlparse(url).path)[1] or ".bin"
    local_path = os.path.join(DOWNLOADS_DIR, key + ext)
    if os.path.exists(local_path):
        return local_path
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    tmp = local_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(resp.content)
    os.replace(tmp, local_path)
    return local_path
