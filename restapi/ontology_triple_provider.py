import logging
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Iterator, List, Optional

try:
    from rdflib import BNode, Graph, Literal, URIRef
    from rdflib.util import guess_format
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("rdflib is required for ontology triple extraction. Please install it via `pip install rdflib`." ) from exc

from ontology_sources import fetch_ontology_resources

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str


class OntologyTripleProvider:
    def __init__(self) -> None:
        self._iterators: Dict[str, Iterator[Triple]] = {}
        self._cache: Dict[str, List[Triple]] = {}

    def _load_triples(self, link: str) -> List[Triple]:
        resources = fetch_ontology_resources(link)
        if not resources:
            logger.warning("No ontology resources found for link %s", link)
            return []
        triples: List[Triple] = []
        graph = Graph()
        for name, data in resources:
            fmt = guess_format(name) or ("xml" if name.lower().endswith(('.owl', '.rdf', '.xml')) else "turtle")
            try:
                graph.parse(data=data.decode("utf-8", errors="ignore"), format=fmt)
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("Failed to parse ontology file %s (%s): %s", name, fmt, exc)

        namespace_mgr = graph.namespace_manager
        for subj, pred, obj in graph:
            if isinstance(subj, BNode) or isinstance(obj, BNode):
                continue
            s = self._term_to_str(namespace_mgr, subj)
            p = self._term_to_str(namespace_mgr, pred)
            o = self._term_to_str(namespace_mgr, obj)
            triples.append(Triple(s, p, o))
        if not triples:
            logger.warning("No triples extracted for link %s", link)
        return triples

    @staticmethod
    def _term_to_str(namespace_mgr, term) -> str:
        if isinstance(term, URIRef):
            try:
                return namespace_mgr.normalizeUri(term)
            except Exception:
                return str(term)
        if isinstance(term, Literal):
            return str(term)
        return str(term)

    def next_triple(self, link: str) -> Optional[Triple]:
        if not link:
            return None
        link = link.strip()
        if link in self._iterators:
            iterator = self._iterators[link]
        else:
            triples = self._load_triples(link)
            if not triples:
                return None
            self._cache[link] = triples
            iterator = cycle(triples)
            self._iterators[link] = iterator
        try:
            return next(iterator)
        except StopIteration:  # pragma: no cover - cycle shouldn't stop
            return None


__all__ = ["OntologyTripleProvider", "Triple"]
