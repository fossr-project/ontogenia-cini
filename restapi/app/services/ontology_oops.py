from typing import Any, Dict, Optional
import os
import re

import requests
from rdflib import Graph


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
    if last_error:
        return ontology_text
    return ontology_text


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
    rdfxml = _ensure_xml_declaration(_to_rdfxml(ontology_text))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<OOPSRequest>"]
    if url_tag and ontology_url:
        lines.append(f"  <{url_tag}>{ontology_url}</{url_tag}>")
    lines.append(f"  <OntologyContent><![CDATA[\n{_wrap_cdata(rdfxml)}\n]]></OntologyContent>")
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
            body = _build_oops_request_xml(
                ontology_text,
                ontology_url=uri_value,
                output_format="RDF/XML",
                url_tag=variant["url_tag"],
            )
            for content_type in content_types:
                status, text = do_request(body, content_type)
                last_status, last_text = status, text
                lower = text.lower()
                if status < 400 and "wrong_execution" not in lower and "unexpected_error" not in lower:
                    return {"raw_response": text, "status_code": status}

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
        return {"raw_response": response.text}
