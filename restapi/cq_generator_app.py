from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Form
import pandas as pd
import openai
import logging
import json
import re
from io import StringIO, BytesIO
import os
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

openai.api_key = os.getenv("OPENAI_API_KEY", "yourkey")

# Optional: ontology parsing (rdflib) and PDF parsing (pypdf)
try:
    from ontology_sources import fetch_ontology_resources
    from rdflib import Graph, BNode, URIRef, Literal
    from rdflib.util import guess_format
    _RDFLIB_AVAILABLE = True
except ImportError:
    _RDFLIB_AVAILABLE = False

try:
    import pypdf
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

# Module-level caches — keyed by URL, populated on first fetch
_ontology_text_cache: dict = {}   # link -> formatted onto_text string
_pdf_text_cache: dict = {}        # link -> extracted pdf_text string
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CQ Generator Service",
    description="Standalone API to generate competency questions from an input CSV with scenario/dataset/description.",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_PATTERNS = [
    {"pattern": "Which [class expression 1][object property expression][class expression 2]?", "example": "Which pizzas contain pork?"},
    {"pattern": "How much does [class expression][datatype property]?", "example": "How much does Margherita Pizza weigh?"},
    {"pattern": "What type of [class expression] is [individual]?", "example": "What type of software is it?"},
    {"pattern": "Is the [class expression 1][class expression 2]?", "example": "Is the software open source?"},
    {"pattern": "What [class expression] has the [numeric modifier][datatype property]?", "example": "What pizza has the lowest price?"},
    {"pattern": "Which are [class expressions]?", "example": "Which are gluten-free bases?"},
]

_INSTRUCTIONS = [
    {
        "instruction": "Do not refer explicitly to dataset variables (e.g., column names, raw codes).",
        "example": {
            "incorrect": "How many cases of COVID-19 in Italy in 2021?",
            "correct": "How many cases of the pathology were registered in the country in a given period?"
        }
    },
    {
        "instruction": "Keep questions simple. Avoid nesting.",
        "example": {
            "incorrect": "What is the capital and population?",
            "correct": ["What is the capital?", "What is the population?"]
        }
    },
    {
        "instruction": "Abstract real entities to generic concepts when needed.",
        "example": {
            "incorrect": "Who is the author of Harry Potter?",
            "correct": "Who is the author of the book?"
        }
    },
    {
        "instruction": "Ensure conceptual diversity. Avoid redundant variations unless conceptually justified.",
        "example": {
            "incorrect": "What is the population affected? / How many people are affected?",
            "correct": "What is the demographic distribution of the affected population?"
        }
    },
]

_CLUSTERING = (
    "Cluster the generated competency questions into thematic areas that reflect distinct conceptual or functional domains. "
    "Each cluster should correspond to a coherent ontological module. Avoid imposing a fixed number of clusters or assigning "
    "each question to a single group. Questions may belong to multiple clusters or may remain unassigned if no meaningful "
    "grouping applies. Use labels meaningful in ontology design, not based on dataset structure."
)

_CONCEPT_EXTRACTION = (
    "Before writing competency questions, do a brief internal conceptual analysis of the provided context "
    "(user stories, dataset description, dataset evidence). "
    "Infer a minimal set of domain elements such as: candidate Classes, Object Properties, Datatype Properties, "
    "and key conceptual dimensions (e.g., Time, Location, Measurement, Agent, Event). "
    "This is NOT ontology generation: do not write OWL, axioms, or a schema; do not output the conceptual analysis. "
    "Use it only to make the competency questions more coherent, diverse, and aligned with the domain."
)

_EXPANDED_PATTERNS = (
    "In addition to the provided competency question patterns, you may use these extended intent categories when justified:\n"
    "- Temporal (time periods, ordering, change over time)\n"
    "- Spatial (location, region, containment)\n"
    "- Aggregation (count, comparison, distribution, highest/lowest)\n"
    "- Relational (associations between classes)\n"
    "- Classification (type/category)\n"
    "If you use an extended intent, put its category name in the 'pattern' field (e.g., 'Temporal')."
)

_ABSTRACTION_CONTROL = (
    "Maintain an appropriate level of abstraction:\n"
    "- Do NOT mention dataset column names, raw codes, or raw identifiers.\n"
    "- Prefer domain-level wording inferred from context (e.g., price, date, location, measurement, category).\n"
    "- Avoid overly generic questions unless the context truly provides nothing more specific.\n"
    "- Avoid overly specific questions that depend on dataset structure or single example values.\n"
)

_TERMINOLOGY_CONSISTENCY = (
    "Ensure terminological consistency across all competency questions:\n"
    "- Identify main domain terms from the user stories and dataset description.\n"
    "- Use the same term consistently.\n"
    "- Avoid switching between near-synonyms unless conceptually justified.\n"
    "- Do not introduce new domain terms not supported by the context.\n"
)

_DEDUPLICATION = (
    "Avoid redundancy:\n"
    "- Do not output paraphrased duplicates.\n"
    "- Do not generate superficial variations unless they target different ontology constructs.\n"
    "- Prioritize conceptual diversity over lexical variation.\n"
)

_PROMPT_STRUCTURE = (
    "Interpret input sources with priority:\n"
    "- User Stories: primary source of intentions and goals.\n"
    "- Dataset Description: semantic framing.\n"
    "- Dataset Profile/Sample: structural evidence to enrich and diversify.\n"
    "- Patterns/Instructions: methodological constraints that must be respected.\n"
    "If there is tension, prioritize domain coherence from User Stories + Description."
)

_CONCEPTUAL_COVERAGE = (
    "Ensure conceptual coverage without forcing a fixed number of questions:\n"
    "- Identify distinct conceptual areas suggested by context "
    "(e.g., Actors, Events, Objects, Measurements, Time, Location, Categories).\n"
    "- Cover multiple areas where meaningful.\n"
    "- Do NOT pad with generic questions.\n"
)

_DATASET_EVIDENCE = (
    "Dataset evidence usage rules:\n"
    "- Treat dataset profile/sample as evidence of the domain only.\n"
    "- Infer high-level concepts and dimensions from it (Time, Location, Measurement, Category, Agent, Event).\n"
    "- Do NOT mention column names, raw identifiers, or specific entity names seen in the sample.\n"
)

_STRUCTURED_OUTPUT = (
    "Output MUST be valid JSON (no markdown, no code fences, no extra text).\n"
    "Use this structure:\n"
    "{\n"
    '  "clusters": [\n'
    "    {\n"
    '      "label": "string",\n'
    '      "rationale": "short explanation (no column names)",\n'
    '      "questions": [\n'
    "        {\n"
    '          "id": "CQ1",\n'
    '          "question": "natural language competency question",\n'
    '          "pattern": "one of the provided patterns OR an extended intent category",\n'
    '          "expected_answer_type": "boolean | entity_list | value | count | min_max | classification",\n'
    '          "notes": "optional short note; may mention intended constructs but not dataset column names"\n'
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ],\n"
    '  "unassigned": [\n'
    "    {\n"
    '      "id": "CQX",\n'
    '      "question": "....",\n'
    '      "pattern": "....",\n'
    '      "expected_answer_type": "....",\n'
    '      "notes": "...."\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Do not aim for a fixed number. Include only semantically meaningful questions aligned with the context."
)

_SYSTEM_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an ontology engineer. Based on the user-provided input, generate competency questions that reflect "
            "relevant conceptual and informational needs. Do not aim for a fixed number—generate only questions that are "
            "semantically meaningful and aligned with the provided context. Follow the patterns and instructions for "
            "clarity, abstraction, and generality."
        ),
    },
    {"role": "system", "content": _DATASET_EVIDENCE},
    {"role": "system", "content": _CONCEPT_EXTRACTION},
    {"role": "system", "content": _EXPANDED_PATTERNS},
    {"role": "system", "content": _ABSTRACTION_CONTROL},
    {"role": "system", "content": _TERMINOLOGY_CONSISTENCY},
    {"role": "system", "content": _DEDUPLICATION},
    {"role": "system", "content": _PROMPT_STRUCTURE},
    {"role": "system", "content": _CONCEPTUAL_COVERAGE},
    {"role": "system", "content": f"Use the following competency question patterns:\n{json.dumps(_PATTERNS)}"},
    {"role": "system", "content": f"Follow these generation instructions:\n{json.dumps(_INSTRUCTIONS)}"},
    {"role": "system", "content": _CLUSTERING},
    {"role": "system", "content": _STRUCTURED_OUTPUT},
]

# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------

def build_dataset_profile(df: pd.DataFrame, max_categories: int = 5) -> str:
    lines = []
    n_rows, n_cols = df.shape
    lines.append(f"- Rows: {n_rows}, Columns: {n_cols}")

    for col in df.columns:
        s = df[col]
        non_null = s.notna().sum()
        null_pct = (1 - non_null / max(n_rows, 1)) * 100
        nunique = s.nunique(dropna=True)
        dtype = str(s.dtype)

        lines.append(f"\nColumn (for analysis only): '{col}'")
        lines.append(f"  - dtype: {dtype}")
        lines.append(f"  - non-null: {non_null} ({100 - null_pct:.1f}%), null%: {null_pct:.1f}%")
        lines.append(f"  - distinct (non-null): {nunique}")

        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().any():
                lines.append(f"  - numeric range: min={s_num.min()}, max={s_num.max()}")
        else:
            if s.dtype == object:
                sample_vals = s.dropna().astype(str).head(20)
                parsed = pd.to_datetime(sample_vals, errors="coerce")
                if len(sample_vals) > 0 and parsed.notna().mean() > 0.7:
                    lines.append("  - hint: values look like dates/times")
            if 0 < nunique <= 50:
                vc = s.dropna().astype(str).value_counts().head(max_categories)
                top = ", ".join([f"{k} ({v})" for k, v in vc.items()])
                lines.append(f"  - top categories: {top}")

        ex = s.dropna().astype(str).head(3).tolist()
        if ex:
            lines.append(f"  - example values: {ex}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _temperature_supported(provider: str, model: str) -> bool:
    """Does a model accept a custom ``temperature`` -- Reasoning / thinking-only models reject it"""
    _OPENAI_NO_TEMP_PREFIXES = ("o1", "o3", "o4")
    _CLAUDE_NO_TEMP_MARKERS = ("opus-4-7", "opus-4-8", "fable", "mythos")
    m = (model or "").lower()
    if any(seg.startswith(_OPENAI_NO_TEMP_PREFIXES) for seg in m.split("/")):
        return False
    if any(t in m for t in _CLAUDE_NO_TEMP_MARKERS):
        return False
    return True


def generate_with_llm(
    messages: list,
    provider: str = "openai",
    model: str = None,
    temperature: Optional[float] = None,
) -> str:
    """Call the specified LLM provider with a full messages list.

    For OpenAI: messages are passed as-is (supports multiple system messages).
    For Claude: system messages are concatenated into the single 'system' field.
    For Together.ai: messages are passed as-is (system messages merged to one).
    """
    if provider == "openai":
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY", "")
            response = openai.chat.completions.create(
                model=model or "gpt-4",
                messages=messages,
                temperature=0,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"Error generating CQ with OpenAI: {e}"

    elif provider == "claude":
        try:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
            system_text = "\n\n".join(
                m["content"] for m in messages if m["role"] == "system"
            )
            user_messages = [m for m in messages if m["role"] != "system"]
            headers = {
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            body = {
                "model": model or "claude-3-opus-20240229",
                "max_tokens": 2000,
                "system": system_text,
                "messages": user_messages,
            }
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
            data = response.json()
            if "content" not in data or not data["content"]:
                logger.error(f"[Claude] Full response: {data}")
                return f"Claude error: {data.get('error', data)}"
            return data["content"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return f"Error generating CQ with Claude: {e}"

    elif provider == "together":
        try:
            together_api_key = os.getenv("TOGETHER_API_KEY", "")
            # Together.ai supports only one system message; merge them
            system_text = "\n\n".join(
                m["content"] for m in messages if m["role"] == "system"
            )
            user_messages = [m for m in messages if m["role"] != "system"]
            merged = [{"role": "system", "content": system_text}] + user_messages
            headers = {
                "Authorization": f"Bearer {together_api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": model or "mistralai/Mistral-7B-Instruct-v0.1",
                "messages": merged,
                "temperature": 0,
                "max_tokens": 2000,
            }
            response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=body)
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Together.ai error: {e}")
            return f"Error generating CQ with Together.ai: {e}"

    elif provider == "ollama":
        try:
            from openai import OpenAI as _OAI
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            client = _OAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="ollama")
            system_text = "\n\n".join(
                m["content"] for m in messages if m["role"] == "system"
            )
            user_messages = [m for m in messages if m["role"] != "system"]
            merged = [{"role": "system", "content": system_text}] + user_messages
            resp = client.chat.completions.create(
                model=model or "llama3.3:70b",
                messages=merged,
                temperature=0,
                max_tokens=1200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error generating CQ with Ollama: {e}"

    elif provider in ("openrouter", "openrouter.ai"):
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return "Error generating CQ with OpenRouter: OPENROUTER_API_KEY is not set."
            
            from openai import OpenAI as _OAI
            # OpenRouter is OpenAI-compatible - same call shape, different base URL.
            client = _OAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            kwargs = dict(model=model or "openai/gpt-4o", messages=messages, max_tokens=4000)
            
            if _temperature_supported("openrouter", model or ""):
                kwargs["temperature"] = 0 if temperature is None else temperature
            elif temperature is not None:
                logger.warning("Model %s does not accept temperature; ignoring %s", model, temperature)
            
            max_retries = 6
            base_delay = 5 
            for attempt in range(max_retries):
                try:
                    resp = client.chat.completions.create(**kwargs)
                    content = resp.choices[0].message.content
                
                    if content is None:
                        raise ValueError("Model returned empty content (NoneType)")
                    return content.strip()
                
                except Exception as e:
                    error_str = str(e)
                    transient_errors = ["429", "rate-limited", "Expecting value", "NoneType", "502", "524"]

                    if any(err in error_str for err in transient_errors):
                        if attempt < max_retries - 1:
                            sleep_time = base_delay * (2 ** (attempt))
                            import time
                            logger.warning(f"OpenRouter rate limit (429). Waiting {sleep_time}s before retry... attempt {attempt + 1}/{max_retries})")
                            time.sleep(sleep_time)
                            continue
                    logger.error(f"OpenRouter error: {e}")
                    return f"Error generating CQ with OpenRouter: {e}"
        
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            return f"Error generating CQ with OpenRouter: {e}"

    else:
        return f"Unknown LLM provider: {provider}"

# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def extract_questions(raw: str) -> str:
    """Extract all questions from the structured JSON response into a single string."""
    try:
        # Strip markdown code fences if present (e.g. ```json ... ```)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        data = json.loads(cleaned)
        questions = []
        for cluster in data.get("clusters", []):
            for q in cluster.get("questions", []):
                text = q.get("question", "").strip()
                if text:
                    questions.append(text if text.endswith("?") else text + "?")
        for q in data.get("unassigned", []):
            text = q.get("question", "").strip()
            if text:
                questions.append(text if text.endswith("?") else text + "?")
        return " ".join(questions) if questions else raw
    except json.JSONDecodeError:
        return raw
    except Exception as e:
        logger.warning("extract_questions fell back to raw text (%s: %s)", type(e).__name__, e)
        return raw

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_dataset_bytes(path: str) -> bytes:
    if path.startswith("https://github.com/") and "/tree/" in path:
        parts = path.split("/tree/")
        repo_url, rest = parts[0], parts[1]
        raw_base = repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        path = f"{raw_base}/{rest}"
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path)
        resp.raise_for_status()
        return resp.content
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Local dataset file not found: {path}")
    with open(path, "rb") as f:
        return f.read()

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/newapi/")
async def generate_cqs_endpoint(
    file: UploadFile = File(...),
    llm_provider: str = Form("openai"),
    model: str = Form(None),
    temperature: Optional[float] = Form(None),
):
    logger.info("Model and provider used for generating cqs → Model: %s, Provider: %s", model, llm_provider)

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        logger.error(f"CSV read error: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading input CSV: {e}")

    def _process_row(args):
        idx, row = args
        scen  = row.get("Scenario", None)
        dpath = row.get("Dataset", None)
        link  = str(row.get("Link", "") or "").strip()
        gold  = row.get("Competency Question", "")
        name  = str(row.get("Name", "") or "").strip()

        has_scen = pd.notna(scen) and str(scen).strip() != ""
        has_data = pd.notna(dpath) and str(dpath).strip() != ""
        has_link = bool(link)

        link_lower = link.lower()
        is_pdf_link  = has_link and link_lower.endswith(".pdf")
        is_onto_link = has_link and not is_pdf_link

        if has_scen and has_data:       mode = "stories+datasets"
        elif has_scen and is_onto_link: mode = "stories+ontologies"
        elif has_data:                  mode = "datasets"
        elif is_onto_link:              mode = "ontologies"
        elif is_pdf_link:               mode = "pdfs"
        elif has_scen:                  mode = "stories"
        else:
            logger.warning(f"Row {idx} skipping; no usable input")
            return None

        prompt_parts = []

        if has_scen:
            prompt_parts.append("SECTION A — USER STORIES (domain intentions)\n" + str(scen))

        if mode in ("datasets", "stories+datasets"):
            sample_csv = profile = ""
            try:
                data_bytes = get_dataset_bytes(str(dpath))
                full_df = pd.read_csv(StringIO(data_bytes.decode("utf-8")))
                sample_csv = full_df.head(10).to_csv(index=False)
                profile = build_dataset_profile(full_df)
            except Exception as e:
                logger.warning(f"Row {idx}: could not load dataset: {e}")
            if profile:
                prompt_parts.append("SECTION C — DATASET PROFILE (analysis only; do not reference columns in questions)\n" + profile)
            if sample_csv:
                prompt_parts.append("SECTION D — DATASET SAMPLE (evidence only; do not copy identifiers/columns)\n" + sample_csv)

        if mode in ("ontologies", "stories+ontologies"):
            # Cache keyed by link — skip fetch+parse for duplicate URLs
            if link not in _ontology_text_cache:
                onto_text = ""
                if _RDFLIB_AVAILABLE:
                    try:
                        resources = fetch_ontology_resources(link)
                        graph = Graph()
                        for fname, fbytes in resources:
                            fmt = guess_format(fname) or ("xml" if fname.lower().endswith((".owl", ".rdf", ".xml")) else "turtle")
                            graph.parse(data=fbytes.decode("utf-8", errors="ignore"), format=fmt)
                        ns = graph.namespace_manager
                        OWL  = "http://www.w3.org/2002/07/owl#"
                        RDFS = "http://www.w3.org/2000/01/rdf-schema#"
                        RDF  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                        classes, props, labels = [], [], {}
                        for s, p, o in graph:
                            ps = str(p)
                            if ps == f"{RDF}type" and str(o) == f"{OWL}Class" and isinstance(s, URIRef):
                                try: classes.append(ns.normalizeUri(s))
                                except: classes.append(str(s).split("#")[-1].split("/")[-1])
                            if ps == f"{RDF}type" and str(o) in (f"{OWL}ObjectProperty", f"{OWL}DatatypeProperty") and isinstance(s, URIRef):
                                try: props.append(ns.normalizeUri(s))
                                except: props.append(str(s).split("#")[-1].split("/")[-1])
                            if ps == f"{RDFS}label" and isinstance(o, Literal):
                                try: labels[ns.normalizeUri(s)] = str(o)
                                except: pass
                        classes = list(dict.fromkeys(classes))[:80]
                        props   = list(dict.fromkeys(props))[:60]
                        parts = []
                        if name: parts.append(f"Ontology: {name}")
                        if classes:
                            parts.append("Classes:\n" + "\n".join(f"  - {c} {('('+labels[c]+')') if c in labels else ''}" for c in classes))
                        if props:
                            parts.append("Properties:\n" + "\n".join(f"  - {p} {('('+labels[p]+')') if p in labels else ''}" for p in props))
                        onto_text = "\n\n".join(parts)
                    except Exception as e:
                        logger.warning(f"Row {idx}: could not parse ontology from {link}: {e}")
                        onto_text = f"Ontology: {name}" if name else ""
                else:
                    onto_text = f"Ontology: {name}" if name else ""
                _ontology_text_cache[link] = onto_text
            else:
                logger.debug(f"Row {idx}: ontology cache hit for {link}")

            onto_text = _ontology_text_cache[link]
            if onto_text:
                prompt_parts.append("SECTION B — ONTOLOGY CONTENT (use classes and properties to infer domain concepts)\n" + onto_text)

        if mode == "pdfs":
            if link not in _pdf_text_cache:
                pdf_text = ""
                if _PYPDF_AVAILABLE:
                    try:
                        raw_url = link
                        if "github.com" in link and "/blob/" in link:
                            raw_url = link.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        resp = requests.get(raw_url.strip(), timeout=30)
                        resp.raise_for_status()
                        reader = pypdf.PdfReader(BytesIO(resp.content))
                        pages_text = [t for page in reader.pages[:10] if (t := page.extract_text())]
                        pdf_text = "\n\n".join(pages_text)[:6000]
                    except Exception as e:
                        logger.warning(f"Row {idx}: could not fetch/parse PDF {link}: {e}")
                _pdf_text_cache[link] = pdf_text
            else:
                logger.debug(f"Row {idx}: PDF cache hit for {link}")

            pdf_text = _pdf_text_cache[link]
            if pdf_text:
                prompt_parts.append("SECTION A — DOCUMENT CONTENT (reference paper/document)\n" + pdf_text)
            elif name:
                prompt_parts.append("SECTION A — DOCUMENT\nTitle: " + name)

        if not prompt_parts:
            logger.warning(f"Row {idx}: no prompt content built for mode={mode}, skipping")
            return None

        user_content = "\n\n".join(prompt_parts).strip()
        messages = _SYSTEM_MESSAGES + [{"role": "user", "content": user_content}]
        import time as _time
        t0 = _time.time()
        raw = generate_with_llm(messages, provider=llm_provider, model=model, temperature=temperature)
        elapsed = _time.time() - t0
        generated = extract_questions(raw)
        n_cqs = len([q for q in generated.split("?") if q.strip()])
        logger.info(f"Row {idx}: generated {n_cqs} CQs in {elapsed:.1f}s (mode={mode})")
        return (idx, gold, generated)

    # Process rows in parallel — I/O-bound (LLM calls + URL fetches)
    MAX_WORKERS = 2
    import time as _time_module
    gen_timestamp = int(_time_module.time())
    gen_results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(gen_results_dir, exist_ok=True)
    gen_partial_path = os.path.join(gen_results_dir, f"generated_cqs_{gen_timestamp}.csv")
    logger.info("Incremental generation saves → %s", gen_partial_path)

    results_map = {}
    partial_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_process_row, (idx, row)): idx for idx, row in df.iterrows()}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                idx, gold, generated = result
                results_map[idx] = (gold, generated)
                partial_rows.append({"_idx": idx, "gold standard": gold, "generated": generated})
                tmp_df = pd.DataFrame(sorted(partial_rows, key=lambda x: x["_idx"])).drop(columns=["_idx"])
                tmp_df.to_csv(gen_partial_path, index=False)

    logger.info("Generation complete. Full CSV saved to %s", gen_partial_path)

    # Reconstruct in original row order
    gold_list = []
    generated_list = []
    for idx in sorted(results_map):
        gold_list.append(results_map[idx][0])
        generated_list.append(results_map[idx][1])

    result_df = pd.DataFrame({
        "gold standard": gold_list,
        "generated": generated_list
    })
    return Response(content=result_df.to_csv(index=False), media_type="text/csv")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cq_generator_app:app", host="127.0.0.1", port=8001, reload=True)
