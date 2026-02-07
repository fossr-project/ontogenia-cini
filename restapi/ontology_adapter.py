import os
import time
import re
import shlex
import subprocess
import tempfile
import unicodedata
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging


class OntologyConstraints(BaseModel):
    output_format: Optional[str] = Field(default="ttl", description="ttl|rdfxml|owl|jsonld")
    iri_base: Optional[str] = None
    naming_policy: Optional[str] = None
    language: Optional[str] = None


class OntologyGenerationRequest(BaseModel):
    system: Optional[str] = Field(
        default=None,
        description="ontogenia|ontogenia-mp|domain-ontogen|neon-gpt|neon-gpt-llms4life",
    )
    dataset_id: Optional[str] = None
    scenario_id: Optional[str] = None
    scenario: Optional[str] = None
    competency_questions: List[str]
    user_stories: Optional[List[str]] = None
    constraints: Optional[OntologyConstraints] = None
    metadata: Optional[Dict[str, Any]] = None
    raw_output: bool = Field(default=False, description="Return raw model output without post-processing")


class OntologyArtifact(BaseModel):
    format: str
    content: str


class OntologyGenerationResponse(BaseModel):
    ontology: OntologyArtifact
    metadata: Dict[str, Any]


app = FastAPI(
    title="Ontology Generation Adapter",
    description="Adapter service exposing POST /generate_ontology for different ontology generation systems.",
    version="1.0.0",
)

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ontology-adapter")


ALLOWED_SYSTEMS = {"ontogenia", "ontogenia-mp", "domain-ontogen", "neon-gpt", "neon-gpt-llms4life"}
_OPENAI_CLIENT: Optional[OpenAI] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _auto_detect_hermit_jar() -> Optional[str]:
    """
    Best-effort local HermiT discovery for the NeOn-GPT pipeline.

    If you place a jar in `<repo>/HermiT/`, the adapter can auto-detect it and run
    HermiT consistency checks without requiring a separate service/container.
    """
    root = _repo_root()
    hermit_dir = root / "HermiT"
    if not hermit_dir.exists():
        return None

    # Prefer the canonical name if present.
    candidates = [
        hermit_dir / "HermiT.jar",
        hermit_dir / "org.semanticweb.HermiT.jar",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    # Fallback: any *.jar in the folder.
    jars = sorted(hermit_dir.glob("*.jar"))
    return str(jars[0]) if jars else None


def _default_prompt_path(system: str) -> Path:
    root = _repo_root()
    if system == "ontogenia":
        return root / "datasets" / "ontology_generation" / "raw" / "ontogenia" / "memoryless_cqbycq_prompt.txt"
    if system == "ontogenia-mp":
        return root / "datasets" / "ontology_generation" / "raw" / "ontogenia" / "ontogenia_mp_paper_prompt.txt"
    if system == "domain-ontogen":
        return root / "datasets" / "ontology_generation" / "raw" / "domain-ontogen" / "prompt.txt"
    if system == "neon-gpt":
        return root / "datasets" / "ontology_generation" / "raw" / "neon-gpt" / "day1_gpt_prompt_list.txt"
    if system == "neon-gpt-llms4life":
        return root / "datasets" / "ontology_generation" / "raw" / "neon-gpt-llms4life" / "prompt.txt"
    raise ValueError(f"Unsupported system: {system}")


def _resolve_prompt_override(system: str, prompt_template: str) -> Optional[Path]:
    """
    Resolve a dataset-provided prompt template to a safe local path.

    The dataset loader stores `prompt_template` under `metadata.prompt_template`.
    We only allow resolving:
      - absolute paths under the repo root
      - filenames under `datasets/ontology_generation/raw/<system>/`
      - for ontogenia/ontogenia-mp, filenames under `datasets/ontology_generation/raw/ontogenia/`
    """
    if not prompt_template:
        return None

    root = _repo_root()
    candidate = Path(prompt_template)
    if candidate.is_absolute():
        try:
            candidate.relative_to(root)
        except Exception:
            raise HTTPException(status_code=400, detail="prompt_template must be under the repository root.")
        return candidate if candidate.is_file() else None

    # Relative filename: treat it as a file under the appropriate raw directory.
    if system in {"ontogenia", "ontogenia-mp"}:
        base = root / "datasets" / "ontology_generation" / "raw" / "ontogenia"
    else:
        base = root / "datasets" / "ontology_generation" / "raw" / system

    candidate = base / prompt_template
    return candidate if candidate.is_file() else None


def _load_prompt_from_path(path: Path) -> str:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Prompt file not found: {path}")
    logger.info("Using prompt file: %s", path)
    raw = path.read_text(encoding="utf-8")
    template = _extract_prompt_template(raw)
    if template != raw:
        logger.info("Extracted prompt template length=%s (from %s)", len(template), len(raw))
    return template


def _load_prompt(system: str) -> str:
    override = os.getenv("ONTOLOGY_PROMPT_FILE", "").strip()
    if override:
        path = Path(override)
    else:
        path = _default_prompt_path(system)
    return _load_prompt_from_path(path)


def _parse_odp_list(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if x is None:
                continue
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
        return out
    return []


def _odps_dir_for_system(system: str) -> Path:
    root = _repo_root()
    # ODP support is currently implemented for Ontogenia prompting.
    return root / "datasets" / "ontology_generation" / "raw" / "ontogenia" / "odps"


def _load_odps_text(system: str, odp_names: List[str]) -> str:
    if not odp_names:
        return ""

    odps_dir = _odps_dir_for_system(system)
    if not odps_dir.exists():
        return ""

    max_chars = int(os.getenv("ONTOLOGY_ODP_MAX_CHARS", "60000"))
    chunks: List[str] = []
    used = 0

    for name in odp_names:
        candidates = [
            odps_dir / name,
            odps_dir / f"{name}.ttl",
            odps_dir / f"{name}.owl",
            odps_dir / f"{name}.rdf",
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if not path:
            logger.warning("ODP not found: %s (looked in %s)", name, odps_dir)
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue

        header = f"# ODP: {path.name}\n"
        block = header + text + "\n"
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)

    return "\n".join(chunks).strip()


def _extract_prompt_template(text: str) -> str:
    if not text:
        return text
    if "```" in text:
        blocks = text.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i].strip()
            if not block:
                continue
            first_line, rest = (block.split("\n", 1) + [""])[:2]
            lang = first_line.strip().lower()
            body = rest.strip() if lang in {"python", "md", "markdown", "text", ""} else block
            if '"""' in body:
                parts = body.split('"""')
                if len(parts) >= 3:
                    body = parts[1]
            if "{CQ}" in body or "{OS}" in body or "{story}" in body:
                return body.strip()
    if '"""' in text:
        parts = text.split('"""')
        if len(parts) >= 3:
            return parts[1].strip()
    return text.strip()


def _story_text(req: OntologyGenerationRequest) -> str:
    if req.scenario:
        return req.scenario
    if req.user_stories:
        return "\n".join(req.user_stories)
    return ""


def _constraints_hint(constraints: Optional[OntologyConstraints]) -> str:
    if not constraints:
        return ""
    parts = []
    if constraints.output_format:
        parts.append(f"output_format={constraints.output_format}")
    if constraints.iri_base:
        parts.append(f"iri_base={constraints.iri_base}")
    if constraints.naming_policy:
        parts.append(f"naming_policy={constraints.naming_policy}")
    if constraints.language:
        parts.append(f"language={constraints.language}")
    if not parts:
        return ""
    return "Constraints: " + "; ".join(parts)


def _use_max_completion_tokens(model: str) -> bool:
    return model.startswith("gpt-5")


def _supports_temperature(model: str) -> bool:
    return not model.startswith("gpt-5")


def _retry_settings() -> tuple[int, float, float]:
    max_retries = int(os.getenv("ONTOLOGY_OPENAI_MAX_RETRIES", "2"))
    base_delay = float(os.getenv("ONTOLOGY_OPENAI_RETRY_BASE_DELAY", "1.0"))
    max_delay = float(os.getenv("ONTOLOGY_OPENAI_RETRY_MAX_DELAY", "10.0"))
    return max_retries, base_delay, max_delay


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")
    timeout = float(os.getenv("ONTOLOGY_OPENAI_TIMEOUT", "60"))
    http2 = os.getenv("ONTOLOGY_OPENAI_HTTP2", "false").lower() == "true"
    disable_keepalive = os.getenv("ONTOLOGY_OPENAI_DISABLE_KEEPALIVE", "false").lower() == "true"
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        if disable_keepalive:
            limits = httpx.Limits(max_keepalive_connections=0, max_connections=1)
        else:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        http_client = httpx.Client(timeout=timeout, http2=http2, limits=limits)
        _OPENAI_CLIENT = OpenAI(api_key=api_key, http_client=http_client)
    return _OPENAI_CLIENT


def _reset_openai_client() -> None:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        close_fn = getattr(_OPENAI_CLIENT, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
    _OPENAI_CLIENT = None


def _should_retry(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_phrases = [
        "connection error",
        "readerror",
        "ssl",
        "tls",
        "timed out",
        "timeout",
        "502",
        "503",
        "504",
        "bad gateway",
        "server error",
    ]
    return any(phrase in message for phrase in retry_phrases)


def _call_openai(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    system_message = os.getenv("OPENAI_SYSTEM_MESSAGE", "").strip()
    max_retries, base_delay, max_delay = _retry_settings()
    for attempt in range(max_retries + 1):
        try:
            client = _get_openai_client()
            logger.info("Calling OpenAI model=%s temperature=%s max_tokens=%s", model, temperature, max_tokens)
            logger.debug("Prompt start\n%s\nPrompt end", prompt)
            if model.startswith("gpt-5"):
                logger.info("Using responses API for model=%s", model)
                request_kwargs = {
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": max_tokens,
                }
                if system_message:
                    request_kwargs["instructions"] = system_message
                response = client.responses.create(**request_kwargs)
                content = _extract_responses_text(response)
            else:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                }
                if _supports_temperature(model):
                    request_kwargs["temperature"] = temperature
                else:
                    logger.info("Temperature omitted for model=%s (uses default)", model)
                if _use_max_completion_tokens(model):
                    request_kwargs["max_completion_tokens"] = max_tokens
                else:
                    request_kwargs["max_tokens"] = max_tokens
                response = client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content.strip()
            logger.info("OpenAI response length=%s", len(content))
            logger.debug("OpenAI response start\n%s\nOpenAI response end", content)
            return content
        except Exception as exc:
            if attempt < max_retries and _should_retry(exc):
                _reset_openai_client()
                delay = min(max_delay, base_delay * (2 ** attempt))
                logger.warning(
                    "OpenAI error: %s. Retrying in %.2fs (%s/%s).",
                    exc,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue
            logger.error("OpenAI error: %s", exc)
            raise HTTPException(status_code=500, detail=f"OpenAI error: {exc}") from exc


def _extract_responses_text(response: Any) -> str:
    text = getattr(response, "output_text", "") or ""
    if text:
        return text
    output = getattr(response, "output", None) or []
    chunks = []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not content:
            continue
        for part in content:
            if isinstance(part, dict):
                part_text = part.get("text") or part.get("refusal")
            else:
                part_text = getattr(part, "text", None) or getattr(part, "refusal", None)
            if part_text:
                chunks.append(part_text)
    if not chunks:
        logger.debug("Responses output had no text content: %s", getattr(response, "output", None))
    return "\n".join(chunks).strip()


def _extract_turtle(text: str) -> str:
    if not text:
        return text
    if "```" in text:
        blocks = text.split("```")
        candidates = []
        for i in range(1, len(blocks), 2):
            block = blocks[i].strip()
            if not block:
                continue
            first_line, rest = (block.split("\n", 1) + [""])[:2]
            lang = first_line.strip().lower()
            if lang in {"turtle", "ttl", "rdf", "owl", "python"}:
                block = rest.strip()
            score = 0
            if "@prefix" in block:
                score += 2
            if "owl:" in block or "rdf:" in block:
                score += 1
            if "Your task is to contribute" in block:
                score -= 3
            if "End of story" in block:
                score -= 2
            if "common mistakes" in block:
                score -= 2
            if "Here is the last RDF" in block:
                score -= 2
            candidates.append((score, block))
        if candidates:
            positive_blocks = [block for score, block in candidates if score > 0]
            if positive_blocks:
                combined = "\n\n".join(block.strip() for block in positive_blocks if block.strip())
                logger.debug("Extracted turtle from %s fenced block(s), length=%s", len(positive_blocks), len(combined))
                return combined.strip()
    idx = text.rfind("@prefix")
    if idx != -1:
        logger.debug("Extracted turtle from last @prefix at index %s", idx)
        return text[idx:].strip()
    return text.strip()


def _clean_turtle(text: str, *, strip_abox: bool = True) -> str:
    if not text:
        return text
    # Strip unicode format characters (can split tokens in LLM output)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    lines = text.splitlines()

    def is_prefix_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("@prefix") and stripped.endswith(".")

    first_prefix = None
    for i, line in enumerate(lines):
        if is_prefix_line(line):
            first_prefix = i
            break
    if first_prefix is not None:
        lines = lines[first_prefix:]

    drop_phrases = [
        "your task is to contribute",
        "is this competency question answerable",
        "end of story",
        "here are some possible mistakes",
        "common mistakes",
        "here is the last rdf",
        "important: before writing",
    ]

    cleaned = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped:
            cleaned.append(line)
            continue
        if lower in {"python", "turtle", "ttl"}:
            continue
        if any(phrase in lower for phrase in drop_phrases):
            continue
        if stripped.startswith("@prefix") and not stripped.endswith("."):
            continue
        # Keep only lines that look like Turtle/OWL content
        if not (
            stripped.startswith(("@prefix", "@base", "#", ":", "_:", "[", "]", "(", ")", ";", ",", ".", "<"))
            or "owl:" in line
            or "rdf:" in line
            or "rdfs:" in line
            or "xsd:" in line
        ):
            continue
        cleaned.append(line)

    # Deduplicate prefix lines while preserving order
    seen_prefixes = set()
    final_lines = []
    for line in cleaned:
        stripped = line.strip()
        if stripped.startswith("@prefix"):
            normalized_prefix = re.sub(r"\s+", " ", stripped)
            if normalized_prefix in seen_prefixes:
                continue
            seen_prefixes.add(normalized_prefix)
        final_lines.append(line)

    if strip_abox:
        final_lines = _strip_instance_blocks(final_lines)
    cleaned_text = "\n".join(final_lines).strip()
    logger.info("Cleaned turtle lines=%s", len(final_lines))
    return cleaned_text


def _strip_instance_blocks(lines: List[str]) -> List[str]:
    if not lines:
        return lines
    keep_types = {
        "owl:Class",
        "owl:ObjectProperty",
        "owl:DatatypeProperty",
        "owl:Ontology",
        "owl:Restriction",
        "rdfs:Class",
        "rdf:Property",
    }
    start_re = re.compile(
        r"^\s*([A-Za-z_][\w-]*:|:)([A-Za-z_][\w-]*)\s+(a|rdf:type)\s+([A-Za-z_][\w-]*:|:)([A-Za-z_][\w-]*)"
    )
    result = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if skipping:
            if stripped.endswith("."):
                skipping = False
            continue
        match = start_re.match(stripped)
        if match:
            obj = f"{match.group(4)}{match.group(5)}"
            if obj not in keep_types:
                skipping = not stripped.endswith(".")
                continue
        if stripped.startswith(":Cl_") and ("owl:ObjectProperty" in stripped or "owl:DatatypeProperty" in stripped):
            if not stripped.endswith("."):
                skipping = True
            continue
        result.append(line)
    return result


def _prefix_bare_identifiers(text: str) -> str:
    if not text:
        return text
    token_re = re.compile(r"(?<![\w:])([A-Za-z_][A-Za-z0-9_-]*)")
    reserved = {"a", "true", "false"}
    iri_re = re.compile(r"<[^>]*>")

    def replace_tokens(segment: str) -> str:
        def repl(match: re.Match) -> str:
            token = match.group(1)
            end = match.end(1)
            if end < len(segment) and segment[end] == ":":
                return token
            if token in reserved:
                return token
            return f":{token}"

        return token_re.sub(repl, segment)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@prefix") or stripped.startswith("@base") or stripped.startswith("#"):
            lines.append(line)
            continue
        parts = line.split('"')
        for i in range(0, len(parts), 2):
            segments = []
            last = 0
            for match in iri_re.finditer(parts[i]):
                segments.append(replace_tokens(parts[i][last:match.start()]))
                segments.append(parts[i][match.start():match.end()])
                last = match.end()
            segments.append(replace_tokens(parts[i][last:]))
            parts[i] = "".join(segments)
        lines.append('"'.join(parts))

    return "\n".join(lines).strip()


def _rewrite_pseudo_owl_constructs(text: str) -> str:
    """
    Fix common pseudo-syntax patterns that LLMs sometimes emit which are not valid
    Turtle, but are close to OWL Functional-style notation.

    Currently supported:
      - owl:AllDisjointClasses( :A :B :C ) .
        -> [] a owl:AllDisjointClasses ; owl:members ( :A :B :C ) .
    """
    if not text:
        return text

    # owl:AllDisjointClasses( :A :B ) .  -> valid Turtle blank node with owl:members list.
    disjoint_pattern = re.compile(r"(?is)owl:AllDisjointClasses\s*\(\s*([^)]+?)\s*\)\s*\.")

    def repl_disjoint(match: re.Match) -> str:
        inner = match.group(1) or ""
        tokens = re.findall(
            r"<[^>]+>|:[A-Za-z_][\w-]*|[A-Za-z_][\w-]*:[A-Za-z_][\w-]*",
            inner,
        )
        if not tokens:
            return match.group(0)
        members = " ".join(tokens)
        return "[] a owl:AllDisjointClasses ;\n    owl:members ( " + members + " ) ."

    return disjoint_pattern.sub(repl_disjoint, text)


def _rewrite_bad_disjointwith_object_lists(text: str) -> str:
    """
    Fix a common Turtle syntax mistake where multiple objects for owl:disjointWith
    are separated by whitespace instead of commas, e.g.:

      :A owl:disjointWith :B :C :D .

    should be:

      :A owl:disjointWith :B, :C, :D .
    """
    if not text:
        return text

    token_re = re.compile(r"<[^>]+>|:[A-Za-z_][\\w-]*|[A-Za-z_][\\w-]*:[A-Za-z_][\\w-]*")
    pattern = re.compile(r"(?im)(\\bowl:disjointWith\\s+)([^;\\.]+?)(\\s*[;\\.])")

    def repl(match: re.Match) -> str:
        head = match.group(1)
        body = match.group(2) or ""
        tail = match.group(3)
        if "," in body:
            return match.group(0)
        if any(x in body for x in ("(", "[", "{")):
            return match.group(0)
        tokens = token_re.findall(body)
        if len(tokens) < 2:
            return match.group(0)
        collapsed_body = re.sub(r"\\s+", " ", body.strip())
        if collapsed_body != " ".join(tokens):
            return match.group(0)
        return head + ", ".join(tokens) + tail

    return pattern.sub(repl, text)


def _fix_trailing_turtle_punctuation(text: str) -> str:
    """
    Fix a common syntax error where the last non-empty Turtle statement ends with
    a `;` or `,` (property/object list continuation) but the model output is
    truncated or missing the following predicate/object, which causes RDFLib to
    fail with errors like:
      - "EOF found when expected verb in property list"

    We conservatively patch only the last non-empty, non-comment line.
    """
    if not text:
        return text
    lines = text.splitlines()
    i = len(lines) - 1
    while i >= 0:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i -= 1
            continue
        line = raw.rstrip()
        match = re.search(r"([;,])(\s*#.*)?\s*$", line)
        if match:
            before = line[: match.start(1)]
            comment = match.group(2) or ""
            lines[i] = before + "." + comment
            return "\n".join(lines).strip()
        break
    return text


def _normalize_turtle(text: str, *, strip_abox: bool = True) -> str:
    cleaned = _clean_turtle(text, strip_abox=strip_abox)
    cleaned = _rewrite_pseudo_owl_constructs(cleaned)
    cleaned = _rewrite_bad_disjointwith_object_lists(cleaned)
    cleaned = _fix_trailing_turtle_punctuation(cleaned)
    normalized = _prefix_bare_identifiers(cleaned)
    return _ensure_prefixes(normalized)


def _syntax_check_turtle(turtle_text: str) -> tuple[bool, Optional[str]]:
    if not turtle_text.strip():
        return False, "Empty ontology content."
    try:
        g = Graph()
        g.parse(data=turtle_text, format="turtle")
        return True, None
    except Exception as exc:
        return False, str(exc)


def _roundtrip_soundness_check_turtle(turtle_text: str) -> tuple[bool, Optional[str]]:
    """
    A lightweight "consistency/soundness" check without a DL reasoner:
    - parse Turtle into an RDF graph
    - serialize to RDF/XML
    - parse RDF/XML back
    This catches some structural/serialization issues beyond raw Turtle parsing.
    """
    try:
        g = Graph()
        g.parse(data=turtle_text, format="turtle")
        xml = g.serialize(format="xml")
        g2 = Graph()
        g2.parse(data=xml, format="xml")
        return True, None
    except Exception as exc:
        return False, str(exc)


def _sanitize_turtle_for_hermit(turtle_text: str) -> tuple[str, List[str]]:
    """
    HermiT is strict about supported datatypes (OWL 2 datatype map) and may reject
    some XSD types (e.g., xsd:gYear). This function applies small deterministic
    substitutions to improve HermiT compatibility.
    """
    mappings = {
        # HermiT often rejects xsd:gYear; represent years as integers instead.
        "xsd:gYear": "xsd:integer",
    }
    replaced: List[str] = []
    out = turtle_text
    for src, dst in mappings.items():
        if src in out:
            out = out.replace(src, dst)
            replaced.append(f"{src}->{dst}")

    # OWLAPI (used by HermiT) will try to load remote imports by default. In our
    # benchmark environment this is undesirable (and may fail due to networking
    # constraints), so we strip *remote* owl:imports statements before invoking
    # HermiT. This is consistent with "reuse by example" (paper) and avoids
    # spurious consistency failures caused by unreachable import IRIs.
    strip_remote_imports = os.getenv("HERMIT_STRIP_REMOTE_IMPORTS", "1").strip().lower() not in {"0", "false", "no", "off"}
    if strip_remote_imports and re.search(r"\bowl:imports\b|<http://www\.w3\.org/2002/07/owl#imports>", out):
        try:
            g = Graph()
            g.parse(data=out, format="turtle")
            removed: List[str] = []
            for subj, obj in list(g.subject_objects(OWL.imports)):
                iri = str(obj)
                if iri.startswith(("http://", "https://")):
                    g.remove((subj, OWL.imports, obj))
                    removed.append(iri)
            if removed:
                out = str(g.serialize(format="turtle"))
                replaced.extend([f"owl:imports removed {iri}" for iri in sorted(set(removed))])
        except Exception:
            # Best-effort only: if parsing fails (unexpected at this stage), fall
            # back to the original text and let HermiT report the issue.
            pass
    return out, replaced


def _add_missing_owl_restriction_types(turtle_text: str, *, strip_abox: bool = True) -> tuple[str, int]:
    """
    OOPS! (and some other tooling) can be fragile when restriction blank nodes are
    missing the explicit `rdf:type owl:Restriction`.

    This function parses the ontology and adds `rdf:type owl:Restriction` to any
    subject that has `owl:onProperty` but is not explicitly typed as a restriction.
    """
    try:
        g = Graph()
        g.parse(data=turtle_text, format="turtle")
    except Exception:
        return turtle_text, 0

    added = 0
    for subj in set(g.subjects(OWL.onProperty, None)):
        if (subj, RDF.type, OWL.Restriction) not in g:
            g.add((subj, RDF.type, OWL.Restriction))
            added += 1

    if not added:
        return turtle_text, 0

    out = g.serialize(format="turtle")
    return _normalize_turtle(str(out), strip_abox=strip_abox), added


def _hermit_consistency_check(turtle_text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Optional consistency check using HermiT.

    Supported modes:
    - HERMIT_MODE=local
      Runs a local Java command: java -jar HERMIT_JAR_PATH -k file://<ontology.rdf>
    - HERMIT_MODE=http
      Calls HERMIT_API_URL with JSON body {"rdfxml": "<...>"} and expects JSON like:
        {"consistent": true|false, "detail": "..."} (additional fields allowed)
    - HERMIT_MODE=docker
      Runs HERMIT_DOCKER_CMD against a temp RDF/XML file.
      HERMIT_DOCKER_CMD may include:
        {file} absolute path to the RDF/XML file
        {dir}  directory containing the file

    If HERMIT_MODE is empty, this returns (True, None, None) and the caller should
    treat it as "skipped".
    """
    mode = os.getenv("HERMIT_MODE", "").strip().lower()
    if not mode:
        # Optional "start everything together" convenience: if enabled, and a local
        # jar is found, behave as HERMIT_MODE=local with an auto-detected jar.
        auto = os.getenv("HERMIT_AUTO", "").strip().lower() in {"1", "true", "yes"}
        jar = _auto_detect_hermit_jar() if auto else None
        if not jar:
            return True, None, None
        mode = "local"
        os.environ.setdefault("HERMIT_JAR_PATH", jar)

    if mode == "http":
        url = os.getenv("HERMIT_API_URL", "").strip()
        if not url:
            return False, "HERMIT_MODE=http but HERMIT_API_URL is empty.", None
        timeout = float(os.getenv("HERMIT_API_TIMEOUT", "60"))
        try:
            g = Graph()
            g.parse(data=turtle_text, format="turtle")
            rdfxml = g.serialize(format="xml")
        except Exception as exc:
            return False, f"Failed to convert Turtle to RDF/XML for HermiT HTTP: {exc}", None
        try:
            resp = httpx.post(url, json={"rdfxml": rdfxml}, timeout=timeout)
        except Exception as exc:
            return False, f"HermiT HTTP call failed: {exc}", None

        raw = (resp.text or "").strip()
        if resp.status_code != 200:
            return False, f"HermiT HTTP status {resp.status_code}: {raw[:1000]}", raw[:4000] if raw else None

        try:
            data = resp.json()
        except Exception:
            return False, f"HermiT HTTP returned non-JSON response: {raw[:1000]}", raw[:4000] if raw else None

        consistent = bool(data.get("consistent", False))
        detail = str(data.get("detail") or data.get("error") or "")
        return (True, None, raw[:4000] if raw else None) if consistent else (False, detail or "Ontology inconsistent.", raw[:4000] if raw else None)

    if mode == "docker":
        cmd_tmpl = os.getenv("HERMIT_DOCKER_CMD", "").strip()
        if not cmd_tmpl:
            return False, "HERMIT_MODE=docker but HERMIT_DOCKER_CMD is empty.", None
        timeout_s = float(os.getenv("HERMIT_DOCKER_TIMEOUT", "120"))
        try:
            with tempfile.TemporaryDirectory(prefix="hermit_check_") as tmpdir:
                # Prefer passing Turtle directly: HermiT/OWLAPI can parse .ttl and this avoids
                # RDF/XML serialization issues in rdflib.
                fpath = Path(tmpdir) / "ontology.ttl"
                fpath.write_text(turtle_text, encoding="utf-8")
                cmd = cmd_tmpl.format(file=str(fpath), dir=str(Path(tmpdir)))
                completed = subprocess.run(
                    shlex.split(cmd),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
                if completed.returncode == 0:
                    return True, None, combined[:4000] if combined else None
                return False, combined[:1500] or f"HermiT docker returned {completed.returncode}", combined[:4000] if combined else None
        except subprocess.TimeoutExpired:
            return False, "HermiT docker check timed out.", None
        except Exception as exc:
            return False, f"HermiT docker check failed: {exc}", None

    if mode == "local":
        jar_path = os.getenv("HERMIT_JAR_PATH", "").strip() or (_auto_detect_hermit_jar() or "")
        if not jar_path:
            return False, "HERMIT_MODE=local but HERMIT_JAR_PATH is empty.", None
        timeout_s = float(os.getenv("HERMIT_LOCAL_TIMEOUT", "120"))
        try:
            with tempfile.TemporaryDirectory(prefix="hermit_check_") as tmpdir:
                # Prefer passing Turtle directly: HermiT/OWLAPI can parse .ttl and this avoids
                # RDF/XML serialization issues in rdflib.
                fpath = Path(tmpdir) / "ontology.ttl"
                fpath.write_text(turtle_text, encoding="utf-8")
                cmd = ["java", "-jar", jar_path, "-k", f"file://{fpath.absolute()}"]
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
                combined_short = combined[:4000] if combined else None
                # Heuristic: mark inconsistent if we see common inconsistency markers.
                if re.search(r"InconsistentOntologyException|inconsistent ontolog", combined, re.IGNORECASE):
                    return False, "Ontology inconsistent (HermiT).", combined_short
                if completed.returncode == 0:
                    return True, None, combined_short
                return False, combined[:1500] or f"HermiT local returned {completed.returncode}", combined_short
        except subprocess.TimeoutExpired:
            return False, "HermiT local check timed out.", None
        except Exception as exc:
            return False, f"HermiT local check failed: {exc}", None

    return False, f"Unsupported HERMIT_MODE={mode!r}. Use 'local', 'http' or 'docker'.", None


def _fix_turtle_with_llm(
    *,
    stage: str,
    ontology_text: str,
    error_message: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    prompt = (
        "You are an expert ontology engineer.\n"
        "Fix the following Turtle ontology.\n\n"
        f"Stage: {stage}\n"
        f"Error/Issue:\n{error_message}\n\n"
        "Rules:\n"
        "- Output ONLY the corrected ontology in valid Turtle syntax.\n"
        "- Do not include explanations, markdown, or code fences.\n"
        "- Keep IRIs/prefixes stable where possible.\n\n"
        "- Use ONLY OWL 2 datatype-map datatypes for ranges (avoid xsd:gYear).\n\n"
        "Ontology:\n"
        f"{ontology_text}\n"
    )
    fixed = _call_openai(prompt, model, temperature, max_tokens)
    return fixed.strip()


def _extract_oops_codes(raw_rdfxml: str) -> List[str]:
    if not raw_rdfxml:
        return []
    # Be tolerant to different namespace prefixes (e.g., oops:, ns1:) and whitespace.
    codes = sorted(
        set(
            re.findall(
                r"<[^:>]+:hasCode[^>]*>\s*(P\d{2})\s*</[^:>]+:hasCode>",
                raw_rdfxml,
            )
        )
    )
    return codes


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _parse_str_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\n]+", value)
        return [p.strip() for p in parts if p and p.strip()]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out
    return []


def _parse_target_metrics(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {"raw": obj}
        except Exception:
            return {"raw": s}
    return {"raw": str(value)}


def _format_bullets(items: List[str]) -> str:
    if not items:
        return ""
    return "\n".join(f"- {x}" for x in items)


def _format_target_metrics(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return ""
    try:
        return json.dumps(metrics, ensure_ascii=False, indent=2)
    except Exception:
        return str(metrics)


def _compute_basic_turtle_counts(ttl: str) -> Dict[str, int]:
    """
    Lightweight local counts (not a full OntoMetrics clone). Used only for
    re-prompting decisions in NeOn-GPT LLMs4Life extended mode.
    """
    try:
        g = Graph()
        g.parse(data=ttl, format="turtle")
    except Exception:
        return {}

    classes = set(g.subjects(RDF.type, OWL.Class))
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    subclass_axioms = list(g.triples((None, RDFS.subClassOf, None)))
    logical_axioms = subclass_axioms  # placeholder: we keep it simple here

    return {
        "triple_count": len(g),
        "class_count": len(classes),
        "object_property_count": len(obj_props),
        "datatype_property_count": len(data_props),
        "subclass_axiom_count": len(subclass_axioms),
        "logical_axiom_count": len(logical_axioms),
    }


def _suspicious_shrink_reason(before: Dict[str, int], after: Dict[str, int]) -> Optional[str]:
    """
    Heuristic guard against LLM "fix" steps that accidentally truncate the ontology
    (e.g., returning only a small snippet that still parses).

    Returns a short reason string when the shrink looks suspicious, else None.
    """
    if not before or not after:
        return None

    before_classes = int(before.get("class_count") or 0)
    after_classes = int(after.get("class_count") or 0)
    before_triples = int(before.get("triple_count") or 0)
    after_triples = int(after.get("triple_count") or 0)

    if before_classes >= 10 and after_classes < max(3, int(before_classes * 0.5)):
        return f"class_count dropped {before_classes}->{after_classes}"
    if before_triples >= 100 and after_triples < max(20, int(before_triples * 0.5)):
        return f"triple_count dropped {before_triples}->{after_triples}"
    if before_triples >= 50 and after_triples < int(before_triples * 0.3):
        return f"triple_count dropped {before_triples}->{after_triples}"

    return None


def _parse_llms4life_categories(value: Any) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - list of {name, keywords}
      - JSON string encoding the list above
    """
    if not value:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            value = json.loads(s)
        except Exception:
            return []
    if not isinstance(value, list):
        return []

    out: List[Dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or raw.get("category") or "").strip()
        keywords = _parse_str_list(raw.get("keywords") or raw.get("terms") or raw.get("items"))
        if not name:
            name = "category"
        out.append({"name": name, "keywords": keywords})
    return out


def _extract_json_array(text: str) -> Optional[str]:
    """
    Best-effort extractor for a JSON array from an LLM response.

    Common failure modes for strict `json.loads`:
    - code fences (```json ... ```)
    - leading/trailing explanation text
    """
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1].strip()


def _auto_categorize_keywords(
    *,
    domain_description: str,
    keywords: List[str],
    category_count: int,
    model: str,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    if not keywords:
        return []
    if category_count <= 1:
        return [{"name": "All", "keywords": keywords}]

    prompt = (
        "You are a domain expert and ontology engineer.\n"
        "Group the following domain keywords into thematic categories to support ontology modularization.\n\n"
        f"Domain description:\n{domain_description}\n\n"
        f"Keywords ({len(keywords)}):\n{_format_bullets(keywords)}\n\n"
        f"Create about {category_count} categories.\n"
        "Output ONLY valid JSON with this shape:\n"
        "[{\"name\": \"CategoryName\", \"keywords\": [\"k1\", \"k2\", ...]}, ...]\n"
        "Rules:\n"
        "- Each keyword must appear in exactly one category.\n"
        "- Keep category names short.\n"
        "- Do not include any text outside JSON.\n"
    )
    raw = _call_openai(prompt, model, temperature, max_tokens)
    try:
        obj = json.loads(raw.strip())
    except Exception:
        extracted = _extract_json_array(raw)
        if not extracted:
            return [{"name": "All", "keywords": keywords}]
        try:
            obj = json.loads(extracted)
        except Exception:
            return [{"name": "All", "keywords": keywords}]

    cats = _parse_llms4life_categories(obj)
    if not cats:
        return [{"name": "All", "keywords": keywords}]

    # Ensure we didn't drop keywords; if we did, fall back to a single category.
    flattened = [k for c in cats for k in _parse_str_list(c.get("keywords"))]
    if len(set(flattened)) != len(set(keywords)):
        return [{"name": "All", "keywords": keywords}]
    return cats


def _merge_turtle_ontologies(ttls: List[str]) -> str:
    merged = Graph()
    for idx, ttl in enumerate(ttls):
        if not ttl:
            continue
        g = Graph()
        try:
            g.parse(data=ttl, format="turtle")
        except Exception as exc:
            # Defensive: category outputs should already be valid Turtle, but if a
            # fragment slipped through with minor syntax issues, try our recovery
            # normalization once before failing the whole request.
            candidate = _normalize_turtle(ttl, strip_abox=False)
            try:
                g.parse(data=candidate, format="turtle")
                ttl = candidate
            except Exception as exc2:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Failed to parse category ontology #{idx + 1} during merge. "
                        f"Original error: {exc}. After normalization: {exc2}."
                    ),
                ) from exc2
        merged += g
    serialized = merged.serialize(format="turtle")
    # IMPORTANT: rdflib already produces syntactically valid Turtle here.
    # Avoid running our generic normalization pipeline (which is designed to
    # recover from messy LLM outputs) because it can accidentally break valid
    # Turtle syntax.
    text = str(serialized).strip()
    # Still strip unicode format characters just in case.
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _render_llms4life_prompt(
    *,
    template: str,
    persona: str,
    domain_name: str,
    domain_description: str,
    category_name: str,
    keywords: List[str],
    target_metrics: Dict[str, Any],
    reuse_examples: str,
    competency_questions: List[str],
) -> str:
    keywords_text = _format_bullets(keywords) or "(none provided)"
    cqs_text = _format_bullets(competency_questions) or "(none)"
    target_metrics_text = _format_target_metrics(target_metrics) or "(none)"
    reuse_text = reuse_examples.strip() if reuse_examples else "(none)"

    prompt = template
    prompt = prompt.replace("{PERSONA}", persona)
    prompt = prompt.replace("{DOMAIN_NAME}", domain_name or "(unspecified)")
    prompt = prompt.replace("{DOMAIN_DESCRIPTION}", domain_description.strip())
    prompt = prompt.replace("{CATEGORY_NAME}", category_name or "All")
    prompt = prompt.replace("{KEYWORDS}", keywords_text)
    prompt = prompt.replace("{TARGET_METRICS}", target_metrics_text)
    prompt = prompt.replace("{REUSE_EXAMPLES}", reuse_text)
    prompt = prompt.replace("{COMPETENCY_QUESTIONS}", cqs_text)
    return prompt.strip()


def _llms4life_paper_prompt_dir() -> Path:
    return (
        _repo_root()
        / "datasets"
        / "ontology_generation"
        / "raw"
        / "neon-gpt-llms4life"
        / "paper_pipeline"
    )


def _load_llms4life_paper_prompt(name: str) -> str:
    path = _llms4life_paper_prompt_dir() / name
    if not path.is_file():
        raise HTTPException(status_code=500, detail=f"Missing NeOn-GPT LLMs4Life prompt template: {path}")
    return path.read_text(encoding="utf-8").strip()


def _render_llms4life_paper_prompt(template: str, mapping: Dict[str, str]) -> str:
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered.strip()


def _parse_bulleted_lines(text: str) -> List[str]:
    if not text:
        return []
    items: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            candidate = line[1:].strip()
        else:
            # Accept simple numbered lists too.
            candidate = re.sub(r"^\d+[\)\.\-]\s*", "", line).strip()
        if candidate:
            items.append(candidate)
    return items


def _ensure_required_items(items: List[str], required: List[str]) -> List[str]:
    required_set = set(required)
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    for rq in required:
        if rq and rq not in seen:
            out.append(rq)
            seen.add(rq)
    # Preserve order but keep the required items exactly as provided.
    out_required = [x for x in out if x in required_set]
    out_other = [x for x in out if x not in required_set]
    return out_required + out_other


def _extract_prefix_block(turtle_text: str) -> str:
    lines: List[str] = []
    for line in (turtle_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(("@prefix", "@base", "PREFIX", "BASE")):
            lines.append(stripped if stripped.endswith(".") else stripped)
            continue
        # Stop once we hit non-prefix content.
        if stripped:
            break
    return "\n".join(lines).strip()


def _split_turtle_prefix_and_body(turtle_text: str) -> tuple[List[str], str]:
    """
    Split a Turtle document into:
    - prefix/base declarations at the top (as stripped lines)
    - the remaining body text (without those declarations)
    """
    if not turtle_text:
        return [], ""
    lines = (turtle_text or "").splitlines()
    prefix_lines: List[str] = []
    body_lines: List[str] = []
    in_prefix = True
    for line in lines:
        stripped = line.strip()
        if in_prefix and stripped.startswith(("@prefix", "@base", "PREFIX", "BASE")):
            prefix_lines.append(stripped)
            continue
        if in_prefix and not stripped:
            # Allow blank lines between prefix/base declarations.
            continue
        in_prefix = False
        body_lines.append(line)
    return prefix_lines, "\n".join(body_lines).strip()


def _merge_prefix_lines(base_prefixes: List[str], fragment_prefixes: List[str]) -> List[str]:
    """
    Merge prefix/base declarations keeping base declarations first and preventing
    duplicates of the same prefix.
    """
    out: List[str] = []
    seen: set[str] = set()

    prefix_re = re.compile(r"(?i)^\s*@prefix\s+([A-Za-z_][\w-]*)?:\s*<[^>]+>\s*\.\s*$")
    base_re = re.compile(r"(?i)^\s*@base\s+<[^>]+>\s*\.\s*$")

    def key_for(line: str) -> str:
        stripped = line.strip()
        if base_re.match(stripped) or stripped.upper().startswith("BASE "):
            return "__base__"
        match = prefix_re.match(stripped)
        if match:
            name = match.group(1) or ""
            return f"{name}:"
        if stripped.upper().startswith("PREFIX "):
            parts = stripped.split()
            if len(parts) >= 2 and parts[1].endswith(":"):
                return parts[1]
        # Fallback: treat the entire line as key.
        return stripped

    for line in base_prefixes + fragment_prefixes:
        stripped = (line or "").strip()
        if not stripped:
            continue
        key = key_for(stripped)
        if key in seen:
            continue
        out.append(stripped)
        seen.add(key)
    return out


def _merge_turtle_with_fragment(*, base_ttl: str, fragment_ttl: str) -> tuple[str, Optional[str]]:
    if not fragment_ttl.strip():
        return base_ttl, None
    # Prefer text-based merge (paper: "Print only the new triples"), because rdflib's
    # Turtle serializer may raise unhelpful errors (e.g., "string index out of range")
    # even when the final ontology can be repaired in the verification stage.
    base_prefixes, base_body = _split_turtle_prefix_and_body(base_ttl)
    frag_prefixes, frag_body = _split_turtle_prefix_and_body(fragment_ttl)
    merged_prefixes = _merge_prefix_lines(base_prefixes, frag_prefixes)

    parts: List[str] = []
    if merged_prefixes:
        parts.append("\n".join(merged_prefixes).strip())
    if base_body:
        parts.append(base_body.strip())
    if frag_body:
        parts.append(frag_body.strip())
    merged = "\n\n".join(p for p in parts if p).strip()

    strict = os.getenv("TURTLE_MERGE_STRICT", "").strip().lower() in {"1", "true", "yes"}
    if strict:
        ok, err = _syntax_check_turtle(merged)
        if not ok:
            return base_ttl, err or "Merged Turtle is invalid."
    return merged, None


def _llms4life_fix_syntax(
    *,
    error_message: str,
    affected_part: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    template = _load_llms4life_paper_prompt("prompt_17_fix_syntax.txt")
    prompt = _render_llms4life_paper_prompt(
        template,
        {
            "ERROR_MESSAGE": error_message.strip(),
            "AFFECTED_PART": affected_part.strip(),
        },
    )
    return _call_openai(prompt, model, temperature, max_tokens).strip()


def _llms4life_fix_inconsistency(
    *,
    error_message: str,
    affected_part: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    template = _load_llms4life_paper_prompt("prompt_18_fix_inconsistency.txt")
    prompt = _render_llms4life_paper_prompt(
        template,
        {
            "ERROR_MESSAGE": error_message.strip(),
            "AFFECTED_PART": affected_part.strip(),
        },
    )
    return _call_openai(prompt, model, temperature, max_tokens).strip()


def _llms4life_fix_pitfall(
    *,
    error_message: str,
    affected_part: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    template = _load_llms4life_paper_prompt("prompt_19_fix_pitfall.txt")
    prompt = _render_llms4life_paper_prompt(
        template,
        {
            "ERROR_MESSAGE": error_message.strip(),
            "AFFECTED_PART": affected_part.strip(),
        },
    )
    return _call_openai(prompt, model, temperature, max_tokens).strip()


def _llms4life_verify_with_tools(
    *,
    ttl: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_syntax_fixes: int,
    max_consistency_fixes: int,
    max_oops_fixes: int,
    keep_abox: bool,
) -> tuple[str, Dict[str, Any]]:
    """
    Verification stage matching the LLMs4Life appendix prompts:
    - RDFLib syntax check (Prompt 17 on failure)
    - HermiT consistency check (Prompt 18 on failure)
    - OOPS pitfall scan/fix (Prompt 19 on failure)
    """
    pipeline_meta: Dict[str, Any] = {
        "pipeline": "llms4life-verification",
        "syntax_fix_attempts": 0,
        "consistency_fix_attempts": 0,
        "oops_fix_attempts": 0,
        "hermit_mode": os.getenv("HERMIT_MODE", "").strip().lower() or None,
        "hermit_raw": None,
        "oops_status_code": None,
        "oops_codes": [],
    }

    strip_abox = not keep_abox

    ttl_candidate = ttl.strip()
    ok, err = _syntax_check_turtle(ttl_candidate)
    while not ok and pipeline_meta["syntax_fix_attempts"] < max_syntax_fixes:
        pipeline_meta["syntax_fix_attempts"] += 1
        fixed = _llms4life_fix_syntax(
            error_message=err or "Turtle parse error.",
            affected_part=ttl_candidate,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ttl_candidate = _normalize_turtle(_extract_turtle(fixed), strip_abox=strip_abox)
        ok, err = _syntax_check_turtle(ttl_candidate)
    if not ok:
        raise HTTPException(status_code=500, detail=f"NeOn-GPT LLMs4Life syntax check failed: {err}")

    hermit_sanitized, hermit_sanitized_replacements = _sanitize_turtle_for_hermit(ttl_candidate)
    if hermit_sanitized_replacements:
        ttl_candidate = hermit_sanitized
        pipeline_meta["hermit_sanitized_replacements"] = hermit_sanitized_replacements

    ok_c, err_c, hermit_raw = _hermit_consistency_check(ttl_candidate)
    pipeline_meta["hermit_raw"] = hermit_raw
    while not ok_c and pipeline_meta["consistency_fix_attempts"] < max_consistency_fixes:
        pipeline_meta["consistency_fix_attempts"] += 1
        fixed = _llms4life_fix_inconsistency(
            error_message=err_c or "Ontology inconsistent per HermiT.",
            affected_part=ttl_candidate,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ttl_candidate = _normalize_turtle(_extract_turtle(fixed), strip_abox=strip_abox)
        hermit_sanitized, hermit_sanitized_replacements = _sanitize_turtle_for_hermit(ttl_candidate)
        if hermit_sanitized_replacements:
            ttl_candidate = hermit_sanitized
            pipeline_meta["hermit_sanitized_replacements"] = hermit_sanitized_replacements
        ok_c, err_c, hermit_raw = _hermit_consistency_check(ttl_candidate)
        pipeline_meta["hermit_raw"] = hermit_raw
    if not ok_c:
        raise HTTPException(status_code=500, detail=f"NeOn-GPT LLMs4Life HermiT consistency check failed: {err_c}")

    oops_url = os.getenv("OOPS_API_URL", "").strip()
    oops_mode = os.getenv("OOPS_API_MODE", "text").strip()
    oops_timeout = int(os.getenv("OOPS_API_TIMEOUT", "60"))
    if oops_url:
        ttl_candidate, restrictions_added = _add_missing_owl_restriction_types(ttl_candidate, strip_abox=strip_abox)
        if restrictions_added:
            pipeline_meta["oops_preprocess_restrictions_added"] = restrictions_added
        try:
            from app.services.ontology_oops import run_oops_scan  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to import OOPS service: {exc}") from exc

        for _ in range(max_oops_fixes + 1):
            oops_result = run_oops_scan(ttl_candidate, oops_url, oops_timeout, oops_mode)
            pipeline_meta["oops_status_code"] = oops_result.get("status_code")
            raw = oops_result.get("raw_response") or ""
            codes = _extract_oops_codes(raw) if oops_result.get("status_code") == 200 else []
            pipeline_meta["oops_codes"] = codes
            if oops_result.get("status_code") != 200 or not codes:
                break
            if pipeline_meta["oops_fix_attempts"] >= max_oops_fixes:
                break

            ttl_checkpoint = ttl_candidate
            pipeline_meta["oops_fix_attempts"] += 1
            fixed = _llms4life_fix_pitfall(
                error_message=raw[:6000] or f"Pitfall codes: {', '.join(codes)}",
                affected_part=ttl_candidate,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ttl_candidate = _normalize_turtle(_extract_turtle(fixed), strip_abox=strip_abox)
            ok_after, err_after = _syntax_check_turtle(ttl_candidate)
            if not ok_after:
                # Best-effort: revert if we broke syntax.
                pipeline_meta["oops_fix_failed_error"] = err_after
                ttl_candidate = ttl_checkpoint
                break
            before_counts = _compute_basic_turtle_counts(ttl_checkpoint)
            after_counts = _compute_basic_turtle_counts(ttl_candidate)
            shrink_reason = _suspicious_shrink_reason(before_counts, after_counts)
            if shrink_reason:
                pipeline_meta["oops_fix_truncation_detected"] = shrink_reason
                ttl_candidate = ttl_checkpoint
                break

    return ttl_candidate, pipeline_meta


def _run_llms4life_paper_pipeline_for_category(
    *,
    domain_name: str,
    domain_description: str,
    persona: str,
    keywords: List[str],
    target_metrics: Dict[str, Any],
    reuse_resource_name: str,
    reuse_resource_description: str,
    reuse_examples: str,
    required_cqs: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    keep_abox: bool,
    max_structure_refinements: int,
) -> tuple[str, Dict[str, Any]]:
    strip_abox = not keep_abox
    meta: Dict[str, Any] = {"pipeline": "neon-gpt-llms4life-paper", "steps": {}}

    mapping_base = {
        "PERSONA": persona,
        "DOMAIN_NAME": domain_name or "(unspecified)",
        "DOMAIN_DESCRIPTION": domain_description.strip(),
        "KEYWORDS": _format_bullets(keywords) or "(none)",
        "TARGET_METRICS": _format_target_metrics(target_metrics) or "(none)",
        "REUSE_RESOURCE_NAME": reuse_resource_name or "(unspecified)",
        "REUSE_RESOURCE_DESCRIPTION": reuse_resource_description or "",
        "REUSE_EXAMPLES": reuse_examples.strip() or "(none)",
        "REQUIRED_CQS": _format_bullets(required_cqs) or "(none)",
        "FEWSHOT_ENTITY_PROPERTY_EXAMPLES": "",
        "FEWSHOT_DATA_PROPERTY_EXAMPLES": "",
        "FEWSHOT_INDIVIDUAL_EXAMPLES": "",
    }

    # Prompt 1: requirements specification
    p1 = _load_llms4life_paper_prompt("prompt_01_requirements.txt")
    prompt1 = _render_llms4life_paper_prompt(p1, mapping_base)
    spec_raw = _call_openai(prompt1, model, temperature, max_tokens).strip()
    meta["steps"]["01_requirements"] = {"output_length": len(spec_raw)}

    # Prompt 2: reuse guidance
    p2 = _load_llms4life_paper_prompt("prompt_02_reuse.txt")
    prompt2 = _render_llms4life_paper_prompt(p2, {**mapping_base})
    reuse_raw = _call_openai(prompt2, model, temperature, max_tokens).strip()
    meta["steps"]["02_reuse"] = {"output_length": len(reuse_raw)}

    # Prompt 3: generate CQs (must include required_cqs)
    p3 = _load_llms4life_paper_prompt("prompt_03_generate_cqs.txt")
    prompt3 = _render_llms4life_paper_prompt(
        p3,
        {
            **mapping_base,
            "REQUIREMENTS_SPEC": spec_raw,
            "REQUIRED_CQS": _format_bullets(required_cqs) or "(none)",
        },
    )
    cqs_raw = _call_openai(prompt3, model, temperature, max_tokens).strip()
    generated_cqs = _parse_bulleted_lines(cqs_raw)
    combined_cqs = _ensure_required_items(generated_cqs, required_cqs)
    meta["steps"]["03_cqs"] = {"generated_count": len(generated_cqs), "combined_count": len(combined_cqs)}

    # Prompt 4: entity/property extraction as JSON
    p4 = _load_llms4life_paper_prompt("prompt_04_extract_entities_properties.txt")
    prompt4 = _render_llms4life_paper_prompt(
        p4,
        {
            **mapping_base,
            "COMPETENCY_QUESTIONS": _format_bullets(combined_cqs) or "(none)",
        },
    )
    ep_raw = _call_openai(prompt4, model, temperature, max_tokens).strip()
    ep_json_text = ep_raw
    ep_obj: Any = None
    ep_parse_error: Optional[str] = None
    try:
        ep_obj = json.loads(ep_raw)
    except Exception:
        extracted = _extract_json_array(ep_raw) or ""
        try:
            ep_obj = json.loads(extracted) if extracted else None
        except Exception as exc:
            ep_parse_error = str(exc)
            ep_obj = [{"cq": cq, "entities": [], "properties": []} for cq in combined_cqs]
    try:
        ep_json_text = json.dumps(ep_obj, ensure_ascii=False, indent=2)
    except Exception:
        ep_json_text = str(ep_obj)
    meta["steps"]["04_entities_properties"] = {"parse_error": ep_parse_error, "output_length": len(ep_raw)}

    # Prompt 5: conceptual model triples
    p5 = _load_llms4life_paper_prompt("prompt_05_conceptual_model_triples.txt")
    prompt5 = _render_llms4life_paper_prompt(
        p5,
        {
            **mapping_base,
            "ENTITY_PROPERTY_JSON": ep_json_text,
        },
    )
    triples_raw = _call_openai(prompt5, model, temperature, max_tokens).strip()
    meta["steps"]["05_triples"] = {"output_length": len(triples_raw)}

    # Prompt 6: base ontology generation
    p6 = _load_llms4life_paper_prompt("prompt_06_generate_ontology.txt")
    prompt6 = _render_llms4life_paper_prompt(
        p6,
        {
            **mapping_base,
            "REQUIREMENTS_SPEC": spec_raw,
            "REUSE_GUIDANCE": reuse_raw,
            "CONCEPTUAL_TRIPLES": triples_raw,
        },
    )
    ontology_raw = _call_openai(prompt6, model, temperature, max_tokens).strip()
    ontology_ttl = _normalize_turtle(_extract_turtle(ontology_raw), strip_abox=strip_abox)
    meta["steps"]["06_ontology"] = {"output_length": len(ontology_raw)}

    # Prompts 7-15: incremental enrichment (new triples) + merge
    incremental_templates = [
        ("07_inverse", "prompt_07_inverse_properties.txt"),
        ("08_reflexive", "prompt_08_reflexive_properties.txt"),
        ("09_symmetric", "prompt_09_symmetric_properties.txt"),
        ("10_functional", "prompt_10_functional_properties.txt"),
        ("11_transitive", "prompt_11_transitive_properties.txt"),
        ("12_data_properties", "prompt_12_data_properties.txt"),
        ("13_individuals", "prompt_13_individuals.txt"),
        ("14_ontology_metadata", "prompt_14_ontology_metadata.txt"),
        ("15_comments", "prompt_15_comments.txt"),
    ]
    for step_key, filename in incremental_templates:
        template = _load_llms4life_paper_prompt(filename)
        prompt = _render_llms4life_paper_prompt(template, {**mapping_base, "ONTOLOGY": ontology_ttl})
        fragment_raw = _call_openai(prompt, model, temperature, max_tokens).strip()
        fragment_ttl = _normalize_turtle(_extract_turtle(fragment_raw), strip_abox=strip_abox)
        merged, merge_err = _merge_turtle_with_fragment(base_ttl=ontology_ttl, fragment_ttl=fragment_ttl)
        ontology_ttl = merged
        meta["steps"][step_key] = {"fragment_length": len(fragment_raw), "merge_error": merge_err}

    # Prompt 16: (optional) structure refinement
    refinement_attempts = 0
    while refinement_attempts < max_structure_refinements:
        refinement_attempts += 1
        counts = _compute_basic_turtle_counts(ontology_ttl)
        class_count = int(counts.get("class_count") or 0)
        subclass_axioms = int(counts.get("subclass_axiom_count") or 0)
        target_class_count = int(target_metrics.get("class_count") or 0) if isinstance(target_metrics, dict) else 0
        needs_refine = False
        if target_class_count and class_count < max(10, int(target_class_count * 0.25)):
            needs_refine = True
        if class_count and subclass_axioms < max(1, class_count - 1):
            needs_refine = True
        if not needs_refine:
            break

        p16 = _load_llms4life_paper_prompt("prompt_16_structure_refinement.txt")
        prompt16 = _render_llms4life_paper_prompt(
            p16,
            {
                **mapping_base,
                "ONTOLOGY": ontology_ttl,
                "REQUIRED_CQS": _format_bullets(required_cqs) or "(none)",
            },
        )
        refined_raw = _call_openai(prompt16, model, temperature, max_tokens).strip()
        refined_candidate = _normalize_turtle(_extract_turtle(refined_raw), strip_abox=strip_abox)
        ok_refine, err_refine = _syntax_check_turtle(refined_candidate)
        if not ok_refine:
            meta["steps"].setdefault("16_refine", []).append({"attempt": refinement_attempts, "discarded_reason": err_refine})
            break
        before_counts = _compute_basic_turtle_counts(ontology_ttl)
        after_counts = _compute_basic_turtle_counts(refined_candidate)
        shrink_reason = _suspicious_shrink_reason(before_counts, after_counts)
        if shrink_reason:
            meta["steps"].setdefault("16_refine", []).append(
                {"attempt": refinement_attempts, "discarded_reason": f"suspicious_shrink: {shrink_reason}"}
            )
            break
        ontology_ttl = refined_candidate
        meta["steps"].setdefault("16_refine", []).append({"attempt": refinement_attempts, "applied": True})

    # Verification (Prompts 17-19 driven fixes)
    verified, verify_meta = _llms4life_verify_with_tools(
        ttl=ontology_ttl,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_syntax_fixes=int(os.getenv("NEON_LIFE_MAX_SYNTAX_FIXES", "3")),
        max_consistency_fixes=int(os.getenv("NEON_LIFE_MAX_CONSISTENCY_FIXES", "3")),
        max_oops_fixes=int(os.getenv("NEON_LIFE_MAX_OOPS_FIXES", "3")),
        keep_abox=keep_abox,
    )
    meta["verification"] = verify_meta
    return verified, meta


def _neon_gpt_pipeline(
    *,
    draft_ttl: str,
    model: str,
    temperature: float,
    max_tokens: int,
    strip_abox: bool = True,
) -> tuple[str, Dict[str, Any]]:
    """
    NeOn-GPT-style pipeline:
    draft -> syntax check/fix loop -> roundtrip soundness check/fix -> (optional) HermiT consistency check/fix -> OOPS scan/fix -> final
    """
    max_syntax_fixes = int(os.getenv("NEON_GPT_MAX_SYNTAX_FIXES", "3"))
    max_soundness_fixes = int(os.getenv("NEON_GPT_MAX_SOUNDNESS_FIXES", "3"))
    max_consistency_fixes = int(os.getenv("NEON_GPT_MAX_CONSISTENCY_FIXES", "3"))
    max_oops_fixes = int(os.getenv("NEON_GPT_MAX_OOPS_FIXES", "3"))

    oops_url = os.getenv("OOPS_API_URL", "").strip()
    oops_mode = os.getenv("OOPS_API_MODE", "text").strip()
    oops_timeout = int(os.getenv("OOPS_API_TIMEOUT", "60"))

    pipeline_meta: Dict[str, Any] = {
        "pipeline": "neon-gpt-with-hermit" if os.getenv("HERMIT_MODE", "").strip() else "neon-gpt-without-hermit",
        "syntax_fix_attempts": 0,
        "soundness_fix_attempts": 0,
        "consistency_fix_attempts": 0,
        "hermit_mode": os.getenv("HERMIT_MODE", "").strip().lower() or None,
        "hermit_raw": None,
        "oops_fix_attempts": 0,
        "oops_status_code": None,
        "oops_codes": [],
    }

    ttl = draft_ttl

    ok, err = _syntax_check_turtle(ttl)
    while not ok and pipeline_meta["syntax_fix_attempts"] < max_syntax_fixes:
        pipeline_meta["syntax_fix_attempts"] += 1
        ttl = _fix_turtle_with_llm(
            stage="syntax_check",
            ontology_text=ttl,
            error_message=err or "Unknown Turtle parse error.",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ttl = _normalize_turtle(_extract_turtle(ttl), strip_abox=strip_abox)
        ok, err = _syntax_check_turtle(ttl)

    if not ok:
        raise HTTPException(status_code=500, detail=f"NeOn-GPT syntax check failed after fixes: {err}")

    ok, err = _roundtrip_soundness_check_turtle(ttl)
    while not ok and pipeline_meta["soundness_fix_attempts"] < max_soundness_fixes:
        pipeline_meta["soundness_fix_attempts"] += 1
        ttl = _fix_turtle_with_llm(
            stage="consistency_check_roundtrip",
            ontology_text=ttl,
            error_message=err or "Roundtrip RDF/XML serialization failed.",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ttl = _normalize_turtle(_extract_turtle(ttl), strip_abox=strip_abox)
        ok, err = _roundtrip_soundness_check_turtle(ttl)

    if not ok:
        raise HTTPException(status_code=500, detail=f"NeOn-GPT soundness check failed: {err}")

    # HermiT can fail on some unsupported datatypes (e.g., xsd:gYear). Apply small
    # deterministic substitutions before invoking HermiT.
    hermit_sanitized, hermit_sanitized_replacements = _sanitize_turtle_for_hermit(ttl)
    if hermit_sanitized_replacements:
        ttl = hermit_sanitized
        pipeline_meta["hermit_sanitized_replacements"] = hermit_sanitized_replacements

    ok, err, hermit_raw = _hermit_consistency_check(ttl)
    pipeline_meta["hermit_raw"] = hermit_raw
    while not ok and pipeline_meta["consistency_fix_attempts"] < max_consistency_fixes:
        pipeline_meta["consistency_fix_attempts"] += 1
        ttl = _fix_turtle_with_llm(
            stage="consistency_check_hermit",
            ontology_text=ttl,
            error_message=err or "Ontology inconsistent per HermiT.",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ttl = _normalize_turtle(_extract_turtle(ttl), strip_abox=strip_abox)

        # Re-apply sanitization in case the fix reintroduced unsupported datatypes.
        hermit_sanitized, hermit_sanitized_replacements = _sanitize_turtle_for_hermit(ttl)
        if hermit_sanitized_replacements:
            ttl = hermit_sanitized
            pipeline_meta["hermit_sanitized_replacements"] = hermit_sanitized_replacements

        ok_s, err_s = _syntax_check_turtle(ttl)
        if not ok_s:
            ttl = _fix_turtle_with_llm(
                stage="post_hermit_syntax_fix",
                ontology_text=ttl,
                error_message=err_s or "Turtle parse error after HermiT fix.",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ttl = _normalize_turtle(_extract_turtle(ttl), strip_abox=strip_abox)
            ok_s, err_s = _syntax_check_turtle(ttl)
            if not ok_s:
                raise HTTPException(status_code=500, detail=f"Syntax broken after HermiT fix: {err_s}")

        ok, err, hermit_raw = _hermit_consistency_check(ttl)
        pipeline_meta["hermit_raw"] = hermit_raw

    if not ok:
        raise HTTPException(status_code=500, detail=f"NeOn-GPT HermiT consistency check failed: {err}")

    # OOPS scan + fix loop (optional; only if OOPS_API_URL configured)
    if oops_url:
        ttl, restrictions_added = _add_missing_owl_restriction_types(ttl, strip_abox=strip_abox)
        if restrictions_added:
            pipeline_meta["oops_preprocess_restrictions_added"] = restrictions_added
        try:
            from app.services.ontology_oops import run_oops_scan  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to import OOPS service: {exc}") from exc

        for _ in range(max_oops_fixes + 1):
            oops_result = run_oops_scan(ttl, oops_url, oops_timeout, oops_mode)
            pipeline_meta["oops_status_code"] = oops_result.get("status_code")
            raw = oops_result.get("raw_response") or ""
            codes = _extract_oops_codes(raw) if oops_result.get("status_code") == 200 else []
            pipeline_meta["oops_codes"] = codes

            # If OOPS fails or reports no pitfalls, stop.
            if oops_result.get("status_code") != 200 or not codes:
                break

            if pipeline_meta["oops_fix_attempts"] >= max_oops_fixes:
                break

            ttl_checkpoint = ttl
            pipeline_meta["oops_fix_attempts"] += 1
            fix_prompt = (
                "You are an expert ontology engineer.\n"
                "The following Turtle ontology has been scanned with OOPS! and pitfalls were found.\n"
                f"Pitfall codes to address: {', '.join(codes)}\n\n"
                "Rules:\n"
                "- Output ONLY the improved ontology in valid Turtle syntax.\n"
                "- Do not include explanations, markdown, or code fences.\n"
                "- Add missing annotations (e.g., rdfs:label/rdfs:comment) when relevant.\n"
                "- Consider adding disjointness axioms when appropriate.\n\n"
                "Ontology:\n"
                f"{ttl}\n"
            )
            fixed = _call_openai(fix_prompt, model, temperature, max_tokens)
            ttl = _normalize_turtle(_extract_turtle(fixed), strip_abox=strip_abox)
            ok, err = _syntax_check_turtle(ttl)
            syntax_fix_tries = 0
            while not ok and syntax_fix_tries < 2:
                syntax_fix_tries += 1
                ttl = _fix_turtle_with_llm(
                    stage="post_oops_syntax_fix",
                    ontology_text=ttl,
                    error_message=err or "Turtle parse error after OOPS fix.",
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                ttl = _normalize_turtle(_extract_turtle(ttl), strip_abox=strip_abox)
                ok, err = _syntax_check_turtle(ttl)

            if not ok:
                # OOPS fixing is best-effort. If we cannot recover valid Turtle, revert
                # to the last known-good ontology and stop trying to fix pitfalls.
                pipeline_meta["oops_fix_failed_error"] = err
                ttl = ttl_checkpoint
                break

            # Guard against fixes that accidentally truncate the ontology while still
            # producing valid Turtle (a common failure mode with LLM "fix" prompts).
            before_counts = _compute_basic_turtle_counts(ttl_checkpoint)
            after_counts = _compute_basic_turtle_counts(ttl)
            shrink_reason = _suspicious_shrink_reason(before_counts, after_counts)
            if shrink_reason:
                pipeline_meta["oops_fix_truncation_detected"] = shrink_reason
                ttl = ttl_checkpoint
                break

    return ttl, pipeline_meta


def _prefix_declared(text: str, prefix: str) -> bool:
    if prefix == ":":
        pattern = r"(?im)^(?:@prefix|prefix)\s*:\s*<[^>]+>\s*\.?"
    else:
        pattern = rf"(?im)^(?:@prefix|prefix)\s+{re.escape(prefix)}:\s*<[^>]+>\s*\.?"
    return re.search(pattern, text) is not None


def _prefix_in_use(text: str, prefix: str) -> bool:
    if prefix == ":":
        return re.search(r"(?m)(^|[\s\[\(\{;,.])(:[A-Za-z_][\w-]*)", text) is not None
    return re.search(rf"\b{re.escape(prefix)}:", text) is not None


def _extract_base_iri(text: str) -> str:
    base_match = re.search(r"(?im)^@base\s*<([^>]+)>", text)
    if not base_match:
        base_match = re.search(r"(?im)^base\s*<([^>]+)>", text)
    if not base_match:
        base_match = re.search(r"<([^>]+)>\s+a\s+owl:Ontology", text)
    if not base_match:
        base_match = re.search(
            r"<([^>]+)>\s+a\s+<http://www\.w3\.org/2002/07/owl#Ontology>",
            text,
        )
    iri = base_match.group(1) if base_match else "http://example.org/ontology#"
    if not iri.endswith(("#", "/")):
        iri = iri.rstrip("/") + "#"
    return iri


def _ensure_prefixes(text: str) -> str:
    if not text:
        return text
    prefix_map = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    }
    base_iri = _extract_base_iri(text)
    prefix_lines = []
    if _prefix_in_use(text, ":") and not _prefix_declared(text, ":"):
        prefix_lines.append(f"@prefix : <{base_iri}> .")
    for prefix, iri in prefix_map.items():
        if _prefix_in_use(text, prefix) and not _prefix_declared(text, prefix):
            prefix_lines.append(f"@prefix {prefix}: <{iri}> .")
    if not prefix_lines:
        return text
    lines = text.splitlines()
    insert_idx = 0
    while insert_idx < len(lines):
        stripped = lines[insert_idx].strip()
        if stripped.startswith(("@prefix", "PREFIX", "@base", "BASE")):
            insert_idx += 1
            continue
        break
    new_lines = lines[:insert_idx] + prefix_lines + lines[insert_idx:]
    return "\n".join(new_lines).strip()


@app.post("/generate_ontology", response_model=OntologyGenerationResponse)
def generate_ontology(req: OntologyGenerationRequest) -> OntologyGenerationResponse:
    if not req.competency_questions:
        raise HTTPException(status_code=400, detail="competency_questions must be a non-empty list.")

    system = (req.system or os.getenv("ONTOLOGY_SYSTEM", "ontogenia")).strip().lower()
    if system not in ALLOWED_SYSTEMS:
        raise HTTPException(status_code=400, detail=f"Invalid ONTOLOGY_SYSTEM: {system}")

    logger.info("Request system=%s dataset_id=%s scenario_id=%s cq_count=%s", system, req.dataset_id, req.scenario_id, len(req.competency_questions))
    story = _story_text(req)
    metadata = req.metadata or {}
    prompt_override = _resolve_prompt_override(system, str(metadata.get("prompt_template") or "").strip())
    prompt_template = _load_prompt_from_path(prompt_override) if prompt_override else _load_prompt(system)
    constraints_hint = _constraints_hint(req.constraints)
    append_constraints = os.getenv("ONTOLOGY_APPEND_CONSTRAINTS", "false").strip().lower() in {"1", "true", "yes"}
    if constraints_hint and not append_constraints:
        logger.info("Constraints provided but ignored (set ONTOLOGY_APPEND_CONSTRAINTS=true to append)")

    model = str(metadata.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    raw_output = bool(req.raw_output or metadata.get("raw_output"))
    if raw_output:
        logger.info("Raw output enabled; skipping Turtle post-processing")

    start = time.time()

    if system == "ontogenia":
        outputs = []
        for cq in req.competency_questions:
            prompt = (
                prompt_template.replace("{story}", story)
                .replace("{CQ}", cq)
                .replace("{rdf}", "")
            )
            if constraints_hint and append_constraints:
                prompt = f"{prompt}\n\n{constraints_hint}"
            raw = _call_openai(prompt, model, temperature, max_tokens)
            if raw_output:
                outputs.append(raw.strip())
            else:
                outputs.append(_normalize_turtle(_extract_turtle(raw)))
        content = "\n\n".join(o for o in outputs if o).strip() if raw_output else _normalize_turtle("\n\n".join(o for o in outputs if o).strip())
    elif system == "ontogenia-mp":
        odp_names = _parse_odp_list((req.metadata or {}).get("odps"))
        odps_text = _load_odps_text(system, odp_names)
        previous = ""
        for cq in req.competency_questions:
            prompt = (
                prompt_template.replace("{story}", story)
                .replace("{CQ}", cq)
                .replace("{rdf}", previous)
                .replace("{odps}", odps_text)
            )
            if constraints_hint and append_constraints:
                prompt = f"{prompt}\n\n{constraints_hint}"
            raw = _call_openai(prompt, model, temperature, max_tokens)
            if raw_output:
                previous = raw.strip()
            else:
                candidate = _normalize_turtle(_extract_turtle(raw))
                ok, err = _syntax_check_turtle(candidate)
                if not ok:
                    candidate = _fix_turtle_with_llm(
                        stage="ontogenia_mp_syntax_fix",
                        ontology_text=candidate,
                        error_message=err or "Turtle parse error.",
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    candidate = _normalize_turtle(_extract_turtle(candidate))
                previous = candidate
        content = previous.strip()
    elif system == "domain-ontogen":
        outputs = []
        for cq in req.competency_questions:
            prompt = prompt_template.replace("{OS}", story).replace("{CQ}", cq)
            if constraints_hint and append_constraints:
                prompt = f"{prompt}\n\n{constraints_hint}"
            raw = _call_openai(prompt, model, temperature, max_tokens)
            if raw_output:
                outputs.append(raw.strip())
            else:
                outputs.append(_normalize_turtle(_extract_turtle(raw)))
        content = "\n\n".join(o for o in outputs if o).strip() if raw_output else _normalize_turtle("\n\n".join(o for o in outputs if o).strip())
    elif system == "neon-gpt-llms4life":
        default_persona = (
            "You are an expert aquatic ecologist and knowledge engineer specializing in developing ecological ontologies. "
            "You have extensive experience in both ecological research and semantic technologies."
        )
        persona = str(metadata.get("persona") or os.getenv("NEON_LIFE_PERSONA", "")).strip() or default_persona
        domain_name = str(metadata.get("domain_name") or metadata.get("domain") or req.dataset_id or "").strip()
        keywords = _parse_str_list(metadata.get("keywords") or metadata.get("keyword_list"))
        target_metrics = _parse_target_metrics(metadata.get("target_metrics") or metadata.get("ontology_metric_counts"))
        reuse_examples = str(metadata.get("reuse_examples") or metadata.get("reuse") or "").strip()
        reuse_resource_name = str(
            metadata.get("reuse_resource_name")
            or metadata.get("reuse_resource")
            or os.getenv("NEON_LIFE_REUSE_RESOURCE_NAME", "ENVO")
        ).strip()
        reuse_resource_description = str(
            metadata.get("reuse_resource_description") or os.getenv("NEON_LIFE_REUSE_RESOURCE_DESCRIPTION", "")
        ).strip()

        # Paper pipeline includes Prompt 13 (individuals), so keep ABox by default.
        keep_abox = True if metadata.get("keep_abox") is None else _boolish(metadata.get("keep_abox"))

        category_mode = _boolish(metadata.get("category_mode"))
        category_count = int(metadata.get("category_count") or os.getenv("NEON_LIFE_CATEGORY_COUNT", "10"))
        categories = _parse_llms4life_categories(metadata.get("categories"))
        if category_mode and not categories and keywords:
            categories = _auto_categorize_keywords(
                domain_description=story,
                keywords=keywords,
                category_count=category_count,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if not categories:
            categories = [{"name": "All", "keywords": keywords}]

        max_structure_refinements = int(
            metadata.get("max_structure_refinements") or os.getenv("NEON_LIFE_MAX_STRUCTURE_REFINEMENTS", "1")
        )

        per_category: List[Dict[str, Any]] = []
        category_ontologies: List[str] = []
        for cat in categories:
            cat_name = str(cat.get("name") or "All").strip() or "All"
            cat_keywords = _parse_str_list(cat.get("keywords")) or keywords
            ttl_cat, cat_meta = _run_llms4life_paper_pipeline_for_category(
                domain_name=f"{domain_name} / {cat_name}".strip(" /"),
                domain_description=story,
                persona=persona,
                keywords=cat_keywords,
                target_metrics=target_metrics,
                reuse_resource_name=reuse_resource_name,
                reuse_resource_description=reuse_resource_description,
                reuse_examples=reuse_examples,
                required_cqs=req.competency_questions,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                keep_abox=keep_abox,
                max_structure_refinements=max_structure_refinements,
            )
            category_ontologies.append(ttl_cat)
            per_category.append({"name": cat_name, "keyword_count": len(cat_keywords), "meta": cat_meta})

        merged = _merge_turtle_ontologies(category_ontologies) if len(category_ontologies) > 1 else category_ontologies[0]
        content, final_verify = _llms4life_verify_with_tools(
            ttl=merged,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_syntax_fixes=int(os.getenv("NEON_LIFE_MAX_SYNTAX_FIXES", "3")),
            max_consistency_fixes=int(os.getenv("NEON_LIFE_MAX_CONSISTENCY_FIXES", "3")),
            max_oops_fixes=int(os.getenv("NEON_LIFE_MAX_OOPS_FIXES", "3")),
            keep_abox=keep_abox,
        )

        pipeline_meta = {
            "pipeline": "neon-gpt-llms4life-paper",
            "category_mode": category_mode,
            "category_count": len(categories),
            "categories": per_category,
            "merged_categories": len(categories) > 1,
            "final_verification": final_verify,
        }
    elif system == "neon-gpt":
        cq_block = "\n".join(f"- {cq}" for cq in req.competency_questions)
        format_hint = req.constraints.output_format if req.constraints else "ttl"
        prompt = (
            f"{prompt_template}\n\n"
            f"Scenario:\n{story}\n\n"
            f"Competency Questions:\n{cq_block}\n\n"
            f"Generate a complete ontology in {format_hint} format. "
            "Output only the ontology."
        )
        if constraints_hint and append_constraints:
            prompt = f"{prompt}\n\n{constraints_hint}"
        raw = _call_openai(prompt, model, temperature, max_tokens)
        if raw_output:
            content = raw.strip()
            pipeline_meta = {}
        else:
            # NeOn-GPT (paper) includes ABox population as a pipeline step; keep ABox.
            draft = _normalize_turtle(_extract_turtle(raw), strip_abox=False)
            content, pipeline_meta = _neon_gpt_pipeline(
                draft_ttl=draft,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                strip_abox=False,
            )

    duration_ms = int((time.time() - start) * 1000)
    ontology_format = req.constraints.output_format if req.constraints else "ttl"
    metadata = {
        "system_name": system,
        "model": model,
        "duration_ms": duration_ms,
    }
    if system in {"neon-gpt", "neon-gpt-llms4life"} and not raw_output:
        metadata["pipeline"] = pipeline_meta
    return OntologyGenerationResponse(
        ontology=OntologyArtifact(format=ontology_format, content=content),
        metadata=metadata,
    )


@app.get("/")
def root() -> Dict[str, Any]:
    hermit_mode = os.getenv("HERMIT_MODE", "").strip().lower() or None
    hermit_auto = os.getenv("HERMIT_AUTO", "").strip().lower() in {"1", "true", "yes"}
    hermit_jar = os.getenv("HERMIT_JAR_PATH", "").strip() or (_auto_detect_hermit_jar() or "")
    return {
        "service": "Ontology Generation Adapter",
        "status": "ok",
        "system": os.getenv("ONTOLOGY_SYSTEM", "ontogenia"),
        "endpoint": "POST /generate_ontology",
        "hermit": {
            "mode": hermit_mode,
            "auto": hermit_auto,
            "jar_detected": bool(hermit_jar),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ontology_adapter:app", host="127.0.0.1", port=8020, reload=False)
