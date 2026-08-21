"""
KEAC CQ Challenge Evaluation Metrics 
====================================
This module provides the core metrics to evaluate generated Competency Questions (CQs) 
against ground truth CQs for the KEAC CQ Challenge. The metrics and overall scoring
function are derived from AskCQ and the Bench4KE hitrate. We refer to the reference
below for a detailed discussion of the metrics and their rationale.

Reference:
Alharbi, R., Tamma, V., Payne, T.R., de Berardinis, J. (2026). A Comparative Study 
of Competency Question Elicitation Methods from Ontology Requirements. In: Acosta, M., 
et al. The Semantic Web. ESWC 2026. Lecture Notes in Computer Science, vol 16549. 
Springer, Cham. https://doi.org/10.1007/978-3-032-25156-5_4

Functions included:
- Sentence-BERT Embedding Generation
- Coverage (Cov) / Hit Rate
- Mean Maximum Similarity (Prec_MMS)
- Average Centroid Distance (ACD)
- Verbosity Penalty (VP)
- Final Ranking Score (S)
"""

import math
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def compute_sentence_embeddings(questions: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Computes sentence embeddings for a list of questions using Sentence-BERT.
    Default model is all-MiniLM-L6-v2 (384-dimensional).
    
    Args:
        questions (List[str]): List of competency questions.
        model_name (str): The sentence-transformers model to use.
        
    Returns:
        np.ndarray: A 2D numpy array of shape (len(questions), embedding_dimension).
    """
    if not questions:
        return np.array([])
        
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("The 'sentence-transformers' package is required to compute embeddings.")
        raise ImportError(
            "The 'sentence-transformers' package is required to compute embeddings. "
            "Please install it using: pip install sentence-transformers"
        )
        
    model = SentenceTransformer(model_name)
    # The encode function automatically handles batching and outputs a numpy array
    embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

def compute_coverage(embeddings_gen: np.ndarray, embeddings_gt: np.ndarray, threshold: float = 0.75) -> float:
    """
    Computes the Coverage (Cov) metric, also known as Hit Rate.
    Measures the proportion of ground truth CQs that have at least one generated CQ 
    exceeding the semantic similarity threshold.
    """
    n_gt = embeddings_gt.shape[0]
    if n_gt == 0:
        return 0.0
    if embeddings_gen.shape[0] == 0:
        return 0.0
        
    cross_similarities = cosine_similarity(embeddings_gt, embeddings_gen)
    max_sims_per_gt = np.max(cross_similarities, axis=1)
    
    hits = np.sum(max_sims_per_gt >= threshold)
    return float(hits / n_gt)

def compute_mean_maximum_similarity(embeddings_gen: np.ndarray, embeddings_gt: np.ndarray) -> float:
    """
    Computes the Mean Maximum Similarity (Prec_MMS).
    Acts as a precision metric by finding the closest ground truth CQ for each 
    generated CQ and computing the average similarity.
    """
    if embeddings_gen.shape[0] == 0 or embeddings_gt.shape[0] == 0:
        return 0.0
        
    cross_similarities = cosine_similarity(embeddings_gen, embeddings_gt)
    max_sims_per_gen = np.max(cross_similarities, axis=1)
    
    return float(np.mean(max_sims_per_gen))

def compute_average_centroid_distance(embeddings: np.ndarray) -> float:
    """
    Computes the Average Centroid Distance (ACD) to measure internal semantic diversity.
    """
    if embeddings.shape[0] == 0:
        return 0.0
        
    centroid = np.mean(embeddings, axis=0)
    distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    
    return float(np.mean(distances_to_centroid))

def compute_verbosity_penalty(n_gen: int, n_gt: int, gamma: float = 1.5) -> float:
    """
    Computes the Verbosity Penalty (VP) to penalise excessive over-generation.
    Uses an exponential decay penalty with a 20% tolerance margin.
    """
    if n_gt == 0:
        return 0.0 if n_gen > 0 else 1.0
        
    ratio = n_gen / n_gt
    penalty_exponent = -gamma * max(0.0, ratio - 1.2)
    
    return math.exp(penalty_exponent)

def compute_challenge_score(
    cqs_gen: List[str], 
    cqs_gt: List[str], 
    threshold: float = 0.75, 
    alpha: float = 0.1, 
    gamma: float = 1.5,
    model_name: str = 'all-MiniLM-L6-v2'
) -> Dict[str, Any]:
    """
    Aggregates Coverage, Prec_MMS, ACD, and the Verbosity Penalty into the final 
    challenge score based on the recommended ranking metric.
    
    Args:
        cqs_gen (List[str]): List of generated Competency Questions.
        cqs_gt (List[str]): List of ground truth Competency Questions.
        threshold (float): Similarity threshold for Coverage (default: 0.75).
        alpha (float): Diversity weight acting as a tie-breaker (default: 0.1).
        gamma (float): Harshness parameter for the Verbosity Penalty (default: 1.5).
        model_name (str): The sentence-transformers model to use.
        
    Returns:
        Dict: A dictionary containing the final score and all individual metric components.
    """
    n_gen = len(cqs_gen)
    n_gt = len(cqs_gt)
    
    if n_gen == 0 or n_gt == 0:
        return {
            "final_score": 0.0, "coverage": 0.0, "prec_mms": 0.0, 
            "acd": 0.0, "verbosity_penalty": 0.0, 
            "n_generated": n_gen, "n_ground_truth": n_gt
        }

    # 1. Compute Embeddings
    embeddings_gen = compute_sentence_embeddings(cqs_gen, model_name=model_name)
    embeddings_gt = compute_sentence_embeddings(cqs_gt, model_name=model_name)
    
    # 2. Compute Base Metrics
    cov = compute_coverage(embeddings_gen, embeddings_gt, threshold)
    prec_mms = compute_mean_maximum_similarity(embeddings_gen, embeddings_gt)
    acd = compute_average_centroid_distance(embeddings_gen)
    vp = compute_verbosity_penalty(n_gen, n_gt, gamma)
    
    # 3. Compute Harmonic Mean of Coverage and Precision (avoid division by zero)
    if (cov + prec_mms) == 0:
        harmonic_mean = 0.0
    else:
        harmonic_mean = 2 * (cov * prec_mms) / (cov + prec_mms)
        
    # 4. Apply Diversity Bonus and Verbosity Penalty
    final_score = harmonic_mean * (1 + alpha * acd) * vp
    
    return {
        "final_score": final_score,
        "coverage": cov,
        "prec_mms": prec_mms,
        "acd": acd,
        "verbosity_penalty": vp,
        "n_generated": n_gen,
        "n_ground_truth": n_gt
    }


# =====================================================================
# TEST SUITE
# =====================================================================

def run_tests():
    """
    Tests the metrics and scoring function on specific corner cases.
    Requires `sentence-transformers` to run successfully.
    """
    print("Running CQ Metrics Corner-Case Tests...\n" + "="*40)
    
    # 1. Exact Match / Full Overlap Case
    print("Test 1: Exact Match (Full Overlap)")
    gt_1 = [
        "What is the capital of France?",
        "How many planets are in the solar system?",
        "Who wrote Romeo and Juliet?"
    ]
    gen_1 = gt_1.copy()
    
    res_1 = compute_challenge_score(gen_1, gt_1)
    print(f"Coverage: {res_1['coverage']} (Expected: 1.0)")
    print(f"Prec_MMS: {res_1['prec_mms']:.4f} (Expected: ~1.0)")
    print(f"VP: {res_1['verbosity_penalty']:.4f} (Expected: 1.0, no over-generation)")
    print(f"Final Score: {res_1['final_score']:.4f}\n")
    assert res_1['coverage'] == 1.0, "Coverage should be 1.0 for exact matches."
    assert res_1['verbosity_penalty'] == 1.0, "VP should be 1.0 since Gen/GT ratio is 1.0 <= 1.2."

    # 2. Rephrasing Case
    print("Test 2: Rephrasing (High Semantic Similarity)")
    gt_2 = [
        "Where is the user's data stored?",
        "What are the system access requirements?",
    ]
    gen_2 = [
        "In what location do you store the data of the user?",
        "Could you list the requirements to access the system?",
    ]
    
    res_2 = compute_challenge_score(gen_2, gt_2)
    print(f"Coverage: {res_2['coverage']} (Expected: 1.0 if sim >= 0.75)")
    print(f"Prec_MMS: {res_2['prec_mms']:.4f} (Expected: High, close to 1.0)")
    print(f"Final Score: {res_2['final_score']:.4f}\n")
    assert res_2['coverage'] > 0.0, "Rephrasings should achieve hits."

    # 3. No Overlap / Completely Irrelevant Case
    print("Test 3: No Overlap (Irrelevant Generation)")
    gt_3 = [
        "How does the authorization process work?",
        "What roles exist in the database?"
    ]
    gen_3 = [
        "Why is the sky blue?",
        "What is the recipe for chocolate cake?"
    ]
    
    res_3 = compute_challenge_score(gen_3, gt_3)
    print(f"Coverage: {res_3['coverage']} (Expected: 0.0)")
    print(f"Prec_MMS: {res_3['prec_mms']:.4f} (Expected: Low)")
    print(f"Final Score: {res_3['final_score']:.4f} (Expected: 0.0 due to harmonic mean)\n")
    assert res_3['coverage'] == 0.0, "Coverage should be 0.0 for completely unrelated text."
    assert res_3['final_score'] == 0.0, "Final score must be 0 if coverage is 0."

    # 4. Verbosity Penalty / Spam Case
    print("Test 4: Over-generation (Triggering VP)")
    gt_4 = ["What is the primary function of the API?"]
    # Generating 5 CQs for 1 GT CQ should trigger the exponential decay penalty
    gen_4 = [
        "What is the primary function of the API?",
        "How do you call the API?",
        "What does the API do?",
        "Is the API fast?",
        "When was the API created?"
    ]
    
    res_4 = compute_challenge_score(gen_4, gt_4)
    print(f"Coverage: {res_4['coverage']} (Expected: 1.0)")
    print(f"VP: {res_4['verbosity_penalty']:.4f} (Expected: < 1.0 due to generating 5x the GT size)")
    print(f"Final Score: {res_4['final_score']:.4f} (Expected: Penalized compared to a concise set)\n")
    assert res_4['verbosity_penalty'] < 1.0, "VP should heavily penalize a ratio of 5.0."

    print("All corner case tests passed successfully!")

if __name__ == "__main__":
    run_tests()