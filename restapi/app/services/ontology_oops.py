from typing import Any, Dict, Optional
import os
import re

import requests
from rdflib import Graph
from rdflib.namespace import RDF, Namespace
from rdflib.term import Node


def _normalize_api_url(api_url: str) -> str:
    if not api_url:
        return api_url
    cleaned = api_url.strip()
    if cleaned.startswith("OOPS_API_URL="):
        cleaned = cleaned.split("=", 1)[1].strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _is_oops_rest_endpoint(api_url: str, mode: str) -> bool:
    if mode == "xml":
        return True
    if not api_url:
        return False
    return api_url.rstrip("/").endswith("/rest")


def _to_rdfxml(ontology_text: str) -> str:
    graph = Graph()
    last_error: Optional[Exception] = None
    for fmt in ("turtle", "xml", "n3", "nt", "json-ld"):
        try:
            graph.parse(data=ontology_text, format=fmt)
            return graph.serialize(format="xml")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Failed to parse ontology for RDF/XML conversion: {last_error}")


def _strip_xml_declaration(text: str) -> str:
    if not text:
        return text
    # Remove UTF-8 BOM and surrounding whitespace before the XML declaration.
    cleaned = text.lstrip("\ufeff").lstrip()
    cleaned = re.sub(r"(?is)^<\?xml[^>]*\?>\s*", "", cleaned)
    return cleaned.lstrip()


_OOPS_NS = Namespace("http://oops.linkeddata.es/def#")


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _parse_oops_rdfxml_summary(raw_rdfxml: str) -> Dict[str, Any]:
    """
    Parse the RDF/XML returned by OOPS! REST and extract a compact summary.

    The benchmark stores the raw RDF/XML response for auditability, but adding a
    small parsed summary makes the output easier to inspect by humans.
    """
    summary: Dict[str, Any] = {
        "pitfall_codes": [],
        "pitfall_instance_counts": {},
        "pitfall_affected_elements_counts": {},
        "pitfalls_total": 0,
        "pitfalls": [],
    }
    if not raw_rdfxml:
        return summary

    graph = Graph()
    graph.parse(data=raw_rdfxml, format="xml")

    def node_to_str(node: Node) -> str:
        return str(node)

    pitfalls = set(graph.subjects(RDF.type, _OOPS_NS.pitfall))
    instance_counts: Dict[str, int] = {}
    affected_counts: Dict[str, int] = {}
    max_sample = int(os.getenv("OOPS_AFFECTED_ELEMENTS_MAX", "50"))

    for pitfall in sorted(pitfalls, key=node_to_str):
        code_node = graph.value(pitfall, _OOPS_NS.hasCode)
        if not code_node:
            continue
        code = str(code_node).strip()
        if not code:
            continue

        importance_node = graph.value(pitfall, _OOPS_NS.hasImportanceLevel)
        importance = str(importance_node).strip() if importance_node else None

        reported_affected = _safe_int(graph.value(pitfall, _OOPS_NS.hasNumberAffectedElements))
        affected_elements = [str(o) for o in graph.objects(pitfall, _OOPS_NS.hasAffectedElement)]
        affected_count = reported_affected if reported_affected is not None else len(affected_elements)

        pit: Dict[str, Any] = {"code": code, "affected_elements_count": int(affected_count or 0)}
        if importance:
            pit["importance"] = importance
        if affected_elements:
            pit["affected_elements_sample"] = affected_elements[:max_sample]
            if len(affected_elements) > max_sample:
                pit["affected_elements_truncated"] = True

        summary["pitfalls"].append(pit)
        instance_counts[code] = instance_counts.get(code, 0) + 1
        affected_counts[code] = affected_counts.get(code, 0) + int(affected_count or 0)

    summary["pitfalls_total"] = len(summary["pitfalls"])
    summary["pitfall_codes"] = sorted(instance_counts.keys())
    summary["pitfall_instance_counts"] = dict(sorted(instance_counts.items()))
    summary["pitfall_affected_elements_counts"] = dict(sorted(affected_counts.items()))
    return summary


def _wrap_cdata(text: str) -> str:
    return text.replace("]]>", "]]]]><![CDATA[>")


def _ensure_xml_declaration(text: str) -> str:
    cleaned = text.lstrip()
    if cleaned.startswith("<?xml"):
        return text
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + text


def _build_oops_request_xml(
    ontology_text: str,
    ontology_url: Optional[str] = None,
    output_format: str = "RDF/XML",
    url_tag: Optional[str] = None,
) -> str:
    # OOPS REST expects RDF/XML inside CDATA. If we include an XML declaration,
    # it must be the first character; otherwise some XML parsers fail with
    # "Content is not allowed in prolog". To avoid this, we strip the XML
    # declaration and embed the RDF/XML starting at <rdf:RDF ...>.
    rdfxml = _strip_xml_declaration(_to_rdfxml(ontology_text)).strip()
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<OOPSRequest>"]
    if url_tag and ontology_url:
        lines.append(f"  <{url_tag}>{ontology_url}</{url_tag}>")
    lines.append(f"  <OntologyContent><![CDATA[{_wrap_cdata(rdfxml)}]]></OntologyContent>")
    lines.append("  <Pitfalls></Pitfalls>")
    lines.append(f"  <OutputFormat>{output_format}</OutputFormat>")
    lines.append("</OOPSRequest>")
    return "\n".join(lines) + "\n"


def run_oops_scan(
    ontology_text: str,
    api_url: str,
    timeout: float,
    mode: str = "text",
    ontology_url: str | None = None,
) -> Dict[str, Any]:
    api_url = _normalize_api_url(api_url)
    if not api_url:
        return {"skipped": True, "reason": "OOPS_API_URL not set"}

    if _is_oops_rest_endpoint(api_url, mode):
        fallback_uri = os.getenv("OOPS_ONTOLOGY_URI", "urn:local:ontology").strip()
        if not fallback_uri:
            fallback_uri = "urn:local:ontology"

        def do_request(body: str, content_type: str) -> tuple[int, str]:
            response = requests.post(
                api_url,
                data=body.encode("utf-8"),
                headers={"Content-Type": content_type},
                timeout=timeout,
            )
            return response.status_code, response.text

        variants = [
            {"url_tag": "OntologyURI", "use_uri": True},
            {"url_tag": "OntologyUrl", "use_uri": True},
            {"url_tag": None, "use_uri": False},
        ]
        content_types = ["text/xml; charset=utf-8", "application/xml"]
        last_status = 0
        last_text = ""

        for variant in variants:
            uri_value = ontology_url if variant["use_uri"] and ontology_url else fallback_uri if variant["use_uri"] else None
            try:
                body = _build_oops_request_xml(
                    ontology_text,
                    ontology_url=uri_value,
                    output_format="RDF/XML",
                    url_tag=variant["url_tag"],
                )
            except Exception as exc:
                # Do not raise: NeOn-GPT pipelines call OOPS during generation
                # and must not crash if conversion fails.
                return {"error": f"Failed to convert ontology to RDF/XML for OOPS: {exc}"}
            for content_type in content_types:
                status, text = do_request(body, content_type)
                last_status, last_text = status, text
                lower = text.lower()
                if status < 400 and "wrong_execution" not in lower and "unexpected_error" not in lower:
                    result: Dict[str, Any] = {"raw_response": text, "status_code": status}
                    try:
                        result.update(_parse_oops_rdfxml_summary(text))
                    except Exception as exc:
                        result["parse_error"] = str(exc)
                    return result

        return {
            "error": f"OOPS REST request failed (status {last_status}).",
            "status_code": last_status,
            "raw_response": last_text,
        }

    if mode == "url":
        if not ontology_url:
            return {"skipped": True, "reason": "OOPS_API_MODE=url requires ontology_url"}
        payload = {"ontologyURL": ontology_url}
        response = requests.post(api_url, data=payload, timeout=timeout)
    elif mode == "file":
        response = requests.post(
            api_url,
            files={"ontology": ("ontology.ttl", ontology_text)},
            timeout=timeout,
        )
    else:
        response = requests.post(
            api_url, data={"ontology": ontology_text}, timeout=timeout
        )

    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        text = response.text
        result: Dict[str, Any] = {"raw_response": text}
        if "<rdf:RDF" in text or "oops.linkeddata.es/def" in text:
            try:
                result.update(_parse_oops_rdfxml_summary(text))
            except Exception as exc:
                result["parse_error"] = str(exc)
        return result
