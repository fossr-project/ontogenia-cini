import os
import time
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
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
    system: Optional[str] = Field(default=None, description="ontogenia|domain-ontogen|neon-gpt")
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


ALLOWED_SYSTEMS = {"ontogenia", "domain-ontogen", "neon-gpt"}
_OPENAI_CLIENT: Optional[OpenAI] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_prompt_path(system: str) -> Path:
    root = _repo_root()
    if system == "ontogenia":
        return root / "datasets" / "ontology_generation" / "raw" / "ontogenia" / "memoryless_cqbycq_prompt.txt"
    if system == "domain-ontogen":
        return root / "datasets" / "ontology_generation" / "raw" / "domain-ontogen" / "prompt.txt"
    if system == "neon-gpt":
        return root / "datasets" / "ontology_generation" / "raw" / "neon-gpt" / "day1_gpt_prompt_list.txt"
    raise ValueError(f"Unsupported system: {system}")


def _load_prompt(system: str) -> str:
    override = os.getenv("ONTOLOGY_PROMPT_FILE", "").strip()
    if override:
        path = Path(override)
    else:
        path = _default_prompt_path(system)
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Prompt file not found: {path}")
    logger.info("Using prompt file: %s", path)
    raw = path.read_text(encoding="utf-8")
    template = _extract_prompt_template(raw)
    if template != raw:
        logger.info("Extracted prompt template length=%s (from %s)", len(template), len(raw))
    return template


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


def _clean_turtle(text: str) -> str:
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
            stripped.startswith(("@prefix", "@base", "#", ":", "_:", "[", "]", ";", ",", ")", ".", "<"))
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


def _normalize_turtle(text: str) -> str:
    cleaned = _clean_turtle(text)
    normalized = _prefix_bare_identifiers(cleaned)
    return _ensure_prefixes(normalized)


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
    prompt_template = _load_prompt(system)
    constraints_hint = _constraints_hint(req.constraints)
    append_constraints = os.getenv("ONTOLOGY_APPEND_CONSTRAINTS", "false").strip().lower() in {"1", "true", "yes"}
    if constraints_hint and not append_constraints:
        logger.info("Constraints provided but ignored (set ONTOLOGY_APPEND_CONSTRAINTS=true to append)")

    metadata = req.metadata or {}
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
    else:  # neon-gpt
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
        content = raw.strip() if raw_output else _normalize_turtle(_extract_turtle(raw))

    duration_ms = int((time.time() - start) * 1000)
    ontology_format = req.constraints.output_format if req.constraints else "ttl"
    metadata = {
        "system_name": system,
        "model": model,
        "duration_ms": duration_ms,
    }
    return OntologyGenerationResponse(
        ontology=OntologyArtifact(format=ontology_format, content=content),
        metadata=metadata,
    )


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "Ontology Generation Adapter",
        "status": "ok",
        "system": os.getenv("ONTOLOGY_SYSTEM", "ontogenia"),
        "endpoint": "POST /generate_ontology",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ontology_adapter:app", host="127.0.0.1", port=8020, reload=False)
