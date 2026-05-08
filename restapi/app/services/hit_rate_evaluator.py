import os
import logging
import numpy as np
import requests
import openai

from sentence_transformers import SentenceTransformer, util as sbert_util

logger = logging.getLogger(__name__)

_sbert_model = None


def _get_sbert_model() -> SentenceTransformer:
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def sbert_similarity_matrix(cqs_a: list, cqs_b: list) -> np.ndarray:
    """Compute cosine similarity matrix between two lists of CQs using SBERT.
    Returns shape (len(cqs_a), len(cqs_b))."""
    model = _get_sbert_model()
    emb_a = model.encode(cqs_a, convert_to_tensor=True)
    emb_b = model.encode(cqs_b, convert_to_tensor=True)
    return sbert_util.cos_sim(emb_a, emb_b).cpu().numpy()


def _call_llm(prompt: str, evaluator_llm: str) -> str:
    """Route LLM call based on model identifier string.

    - Identifiers starting with 'gpt' → OpenAI API (OPENAI_API_KEY)
    - Identifiers containing 'claude' → Anthropic API (ANTHROPIC_API_KEY)
    - Everything else → Together.ai (TOGETHER_API_KEY)
    """
    if evaluator_llm.startswith("gpt"):
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        response = openai.chat.completions.create(
            model=evaluator_llm,
            messages=[
                {"role": "system", "content": "You are an ontology engineering expert specialising in competency questions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    elif "claude" in evaluator_llm:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": evaluator_llm,
            "max_tokens": 600,
            "system": "You are an ontology engineering expert specialising in competency questions.",
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()

    elif evaluator_llm.startswith("ollama/"):
        # Ollama local/remote — model string format: "ollama/<model_name>"
        model_name = evaluator_llm[len("ollama/"):]
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        from openai import OpenAI
        client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="ollama")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an ontology engineering expert specialising in competency questions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()

    else:
        # Together.ai (Mistral, LLaMA, etc.)
        api_key = os.getenv("TOGETHER_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": evaluator_llm,
            "messages": [
                {"role": "system", "content": "You are an ontology engineering expert specialising in competency questions."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 600,
        }
        resp = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def _generate_gap_cqs(
    missed_bench_cqs: list,
    covered_bench_cqs: list,
    scenario_context: str,
    evaluator_llm: str,
    k: int,
) -> list:
    """Ask the evaluator LLM to generate up to k CQs per missed benchmark CQ.

    Uses the scenario/domain context (same input as the external tool), not the
    missed CQs themselves, so the evaluation is not circular.
    """
    covered_str = "\n".join(f"- {c}" for c in covered_bench_cqs) if covered_bench_cqs else "None"
    missed_str = "\n".join(f"- {m}" for m in missed_bench_cqs)
    total = k * len(missed_bench_cqs)

    prompt = (
        f"You are evaluating a knowledge engineering tool against a benchmark.\n\n"
        f"Domain/Scenario context:\n{scenario_context}\n\n"
        f"The following benchmark competency questions were NOT covered by the tool:\n{missed_str}\n\n"
        f"Already covered CQs (do NOT suggest questions similar to these):\n{covered_str}\n\n"
        f"Task: Generate up to {total} competency questions that address the uncovered areas above.\n"
        f"- Rank them by relevance (most relevant first)\n"
        f"- Do not suggest questions already covered by existing ones\n"
        f"- Only suggest genuinely new questions not yet considered\n"
        f"- Output one question per line, ending with '?', no numbering or bullets"
    )

    try:
        raw = _call_llm(prompt, evaluator_llm)
    except Exception as e:
        logger.error(f"Evaluator LLM call failed: {e}")
        return []

    lines = [l.strip() for l in raw.split("\n") if l.strip() and "?" in l]
    return lines[:total]


class HitRateEvaluator:
    """Two-phase benchmark coverage evaluator.

    Phase 1 — base coverage:
        For each benchmark CQ, find its best-matching tool CQ via SBERT cosine
        similarity. Count covered (>= threshold) and missed (< threshold).

    Phase 2 — LLM gap filling:
        Ask an independent evaluator LLM (must differ from the tool's LLM) to
        generate candidate CQs for the missed area using the domain context.
        Keep only candidates that match at least one benchmark CQ (>= threshold).
        Count how many previously missed benchmark CQs are now recovered.

    Bonus — rescued tool CQs:
        Tool CQs that matched no benchmark CQ in Phase 1 but do match the
        evaluator's gap proposals are flagged as potentially useful questions
        not yet captured by the benchmark.
    """

    def __init__(self, threshold: float = 0.6, k: int = 3):
        self.threshold = threshold
        self.k = k

    def compute(
        self,
        bench_cqs: list,
        tool_cqs: list,
        scenario_context: str,
        evaluator_llm: str,
        tool_llm: str = None,
    ) -> dict:
        if not bench_cqs:
            raise ValueError("bench_cqs cannot be empty.")
        if not tool_cqs:
            raise ValueError("tool_cqs cannot be empty.")
        if tool_llm and tool_llm.strip().lower() == evaluator_llm.strip().lower():
            raise ValueError(
                f"evaluator_llm must differ from tool_llm (both are '{tool_llm}'). "
                "Using the same LLM for generation and evaluation introduces bias."
            )

        # --- Phase 1: base coverage ---
        sim_matrix = sbert_similarity_matrix(bench_cqs, tool_cqs)  # (|Bench|, |Tool|)
        best_per_bench = sim_matrix.max(axis=1)  # (|Bench|,)
        covered_mask = best_per_bench >= self.threshold

        covered = int(covered_mask.sum())
        missed = int((~covered_mask).sum())
        covered_bench_cqs = [bench_cqs[i] for i in range(len(bench_cqs)) if covered_mask[i]]
        missed_bench_cqs = [bench_cqs[i] for i in range(len(bench_cqs)) if not covered_mask[i]]
        coverage_rate = covered / len(bench_cqs)

        # --- Phase 2: LLM gap filling ---
        recovered = 0
        valid_gap_cqs = []
        rescued_tool_cqs = []

        if missed_bench_cqs:
            gap_cqs = _generate_gap_cqs(
                missed_bench_cqs, covered_bench_cqs, scenario_context, evaluator_llm, self.k
            )

            if gap_cqs:
                # Keep gap CQs that match at least one benchmark CQ
                gap_vs_bench = sbert_similarity_matrix(bench_cqs, gap_cqs)  # (|Bench|, |gap|)
                gap_max_vs_bench = gap_vs_bench.max(axis=0)  # (|gap|,)
                valid_mask = gap_max_vs_bench >= self.threshold
                valid_gap_cqs = [gap_cqs[i] for i in range(len(gap_cqs)) if valid_mask[i]]

                if valid_gap_cqs:
                    # Count missed bench CQs recovered by valid gap CQs
                    missed_vs_valid = sbert_similarity_matrix(missed_bench_cqs, valid_gap_cqs)
                    recovered = int((missed_vs_valid.max(axis=1) >= self.threshold).sum())

                    # Rescue: tool CQs that had no bench match but match valid gap CQs
                    uncovered_tool_mask = sim_matrix.max(axis=0) < self.threshold
                    uncovered_tool_cqs = [tool_cqs[i] for i in range(len(tool_cqs)) if uncovered_tool_mask[i]]
                    if uncovered_tool_cqs:
                        unlisted_vs_gap = sbert_similarity_matrix(uncovered_tool_cqs, valid_gap_cqs)
                        rescued_mask = unlisted_vs_gap.max(axis=1) >= self.threshold
                        rescued_tool_cqs = [
                            uncovered_tool_cqs[i]
                            for i in range(len(uncovered_tool_cqs))
                            if rescued_mask[i]
                        ]

        augmented_coverage_rate = (covered + recovered) / len(bench_cqs)

        return {
            "coverage_rate": round(float(coverage_rate), 4),
            "augmented_coverage_rate": round(float(augmented_coverage_rate), 4),
            "covered": covered,
            "missed": missed,
            "recovered": recovered,
            "threshold": self.threshold,
            "evaluator_llm_used": evaluator_llm,
            "tool_llm_declared": tool_llm,
            "missed_bench_cqs": missed_bench_cqs,
            "gap_cqs_proposed": valid_gap_cqs,
            "rescued_tool_cqs": rescued_tool_cqs,
        }
