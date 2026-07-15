import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import requests
from urllib.parse import urlparse
import math
from typing import Optional

from app.utils.llm_clients import get_llm_client

try:
    from app.services.heatmap_generator import generate_heatmap, save_heatmap_image
except Exception:
    def generate_heatmap(*args, **kwargs):
        return None
    def save_heatmap_image(*args, **kwargs):
        return None

try:
    from app.config import OPENAI_API_KEY
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_AVAILABLE = True
except Exception:
    _BERTSCORE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _BLEU_AVAILABLE = True
    _smooth = SmoothingFunction().method1
except Exception:
    _BLEU_AVAILABLE = False
    def sentence_bleu(*args, **kwargs):
        return 0.0
    _smooth = None

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except Exception:
    _ROUGE_AVAILABLE = False


class CQValidator:
    def __init__(self, output_folder: str, model: str = "gpt-4", validation_mode: str = "all", provider: Optional[str] = None):
        self.output_folder = output_folder
        self.model = model
        self.validation_mode = validation_mode
        self.provider = provider    # Optional LLM provider for the factory design pattern; 'None' preserves prior behavior
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._rouge = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True) if _ROUGE_AVAILABLE else None
        self._embedding_cache: dict = {}
        self._embedding_dim = self.sbert_model.get_sentence_embedding_dimension()
        self._download_warnings = []

    @staticmethod
    def remove_html_tags(text: str) -> str:
        return re.sub(r'<[^>]+>', '', text)

    def _encode_with_cache(self, sentences: list) -> np.ndarray:
        if not sentences:
            return np.empty((0, self._embedding_dim))
        missing = [s for s in sentences if s not in self._embedding_cache]
        if missing:
            new_embs = self.sbert_model.encode(missing, convert_to_numpy=True)
            for sent, emb in zip(missing, new_embs):
                self._embedding_cache[sent] = emb
        return np.stack([self._embedding_cache[s] for s in sentences])

    @staticmethod
    def _best_match_vector(cosine_sim_matrix: np.ndarray) -> np.ndarray:
        return cosine_sim_matrix.max(axis=1) if cosine_sim_matrix.size else np.array([])

    @staticmethod
    def _precision_at_threshold(best_vec: np.ndarray, thr: float = 0.6) -> float:
        if best_vec.size == 0:
            return 0.0
        return float((best_vec >= thr).sum()) / float(best_vec.size)

    @staticmethod
    def _matches_at_threshold(best_vec: np.ndarray, thr: float = 0.6) -> int:
        if best_vec.size == 0:
            return 0
        return int((best_vec >= thr).sum())

    @staticmethod
    def _best_match_per_gold(cosine_sim_matrix: np.ndarray) -> np.ndarray:
        if cosine_sim_matrix.size == 0:
            return np.array([])
        return cosine_sim_matrix.max(axis=0)

    @staticmethod
    def _classify_link(url: str) -> str:
        if not isinstance(url, str) or not url.strip():
            return 'other'
        u = url.lower()
        if any(u.endswith(ext) for ext in ('.owl', '.ttl', '.rdf', '.xml', '.n3', '.nt')):
            return 'ontology'
        if u.endswith('.pdf') or 'pdf' in u:
            return 'pdf'
        return 'other'

    def _safe_download(self, url: str, dest_folder: str, timeout: int = 10) -> Optional[str]:
        os.makedirs(dest_folder, exist_ok=True)
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return None
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            filename = os.path.basename(parsed.path) or f"download_{abs(hash(url))}"
            out_path = os.path.join(dest_folder, filename)
            with open(out_path, 'wb') as f:
                f.write(r.content)
            return out_path
        except Exception as e:
            self._download_warnings.append(f"Failed to download {url}: {e}")
            return None

    def generate_response(self, chosen_model: str, messages, max_tokens: int = 3000, temperature: float = 0) -> str:
        client = get_llm_client(self.provider)
        return client.chat_completion(messages=messages, model=chosen_model, max_tokens=max_tokens, temperature=temperature)

    def aggregate_dataframe_metrics(self, df: pd.DataFrame, gold_col: str = None,
                                    generated_col: str = 'generated', link_cols: list = None,
                                    output_ontologies_folder: str = None,
                                    precision_threshold: float = 0.6,
                                    compute_pairwise_metrics: bool = False) -> dict:
        if gold_col is None:
            if 'gold standard' in df.columns:
                gold_col = 'gold standard'
            elif 'Competency Question' in df.columns:
                gold_col = 'Competency Question'
            else:
                raise ValueError("No gold column found.")

        candidate_links = link_cols or [c for c in ('Link', 'Links', 'ontologies', 'ontology', 'ontologies_list') if c in df.columns]
        group_keys = ['Project Name'] + (['Dataset'] if 'Dataset' in df.columns else [])
        grouped = df.groupby(group_keys)

        projects_out = {}
        overall_rows = []

        for gname, group in grouped:
            key = gname if isinstance(gname, tuple) else (gname, '')
            golds = list(dict.fromkeys(group[gold_col].dropna().astype(str).str.strip().unique().tolist()))
            gens = list(dict.fromkeys(group[generated_col].dropna().astype(str).str.strip().unique().tolist())) if generated_col in group.columns else []

            ontologies, pdfs = [], []
            for c in candidate_links:
                if c in group.columns:
                    for cell in group[c].dropna().astype(str):
                        for p in re.split(r"[,;]", cell):
                            p = p.strip().strip("[]'\"")
                            if not p:
                                continue
                            cls = self._classify_link(p)
                            if cls == 'ontology' or any(x in p.lower() for x in ('.owl', '.ttl')):
                                ontologies.append(p)
                            elif cls == 'pdf':
                                pdfs.append(p)

            downloaded = []
            if output_ontologies_folder and ontologies:
                for url in ontologies:
                    path = self._safe_download(url, output_ontologies_folder)
                    if path:
                        downloaded.append(path)

            metrics = {}
            if golds and gens:
                try:
                    embeddings = self._encode_with_cache(gens + golds)
                    cosine_sim_matrix = cosine_similarity(embeddings[:len(gens)], embeddings[len(gens):])
                    per_gen_best = self._best_match_vector(cosine_sim_matrix)
                    per_gold_best = self._best_match_per_gold(cosine_sim_matrix)

                    metrics['num_gold'] = len(golds)
                    metrics['num_generated'] = len(gens)
                    metrics['mean_cosine_all_pairs'] = float(np.mean(cosine_sim_matrix))
                    metrics['median_cosine_all_pairs'] = float(np.median(cosine_sim_matrix))
                    metrics['std_cosine_all_pairs'] = float(np.std(cosine_sim_matrix))
                    metrics['per_generated_best_mean'] = float(np.mean(per_gen_best)) if per_gen_best.size else None
                    metrics['per_generated_best_median'] = float(np.median(per_gen_best)) if per_gen_best.size else None
                    metrics['per_gold_best_mean'] = float(np.mean(per_gold_best)) if per_gold_best.size else None
                    metrics['per_gold_best_median'] = float(np.median(per_gold_best)) if per_gold_best.size else None
                    metrics['precision_at_thr_generated'] = float(self._precision_at_threshold(per_gen_best, thr=precision_threshold))
                    metrics['precision_at_thr_gold'] = float(self._precision_at_threshold(per_gold_best, thr=precision_threshold))
                    metrics['matches_at_thr_generated'] = self._matches_at_threshold(per_gen_best, thr=precision_threshold)
                    metrics['matches_at_thr_gold'] = self._matches_at_threshold(per_gold_best, thr=precision_threshold)
                    metrics['per_generated_best_vector'] = per_gen_best.tolist()
                    metrics['per_gold_best_vector'] = per_gold_best.tolist()
                except Exception as e:
                    metrics['error'] = f"embedding/cosine computation failed: {e}"
            else:
                metrics['num_gold'] = len(golds)
                metrics['num_generated'] = len(gens)

            if compute_pairwise_metrics and golds and gens and 'error' not in metrics:
                try:
                    n_gen, n_gold = len(gens), len(golds)
                    bert_f1_matrix = np.zeros((n_gen, n_gold))
                    bleu_matrix = np.zeros((n_gen, n_gold))
                    rougeL_matrix = np.zeros((n_gen, n_gold))

                    for i, cq_gen in enumerate(gens):
                        for j, cq_man in enumerate(golds):
                            if _BERTSCORE_AVAILABLE:
                                try:
                                    _, _, F1 = bert_score_fn([cq_gen], [cq_man], lang='en',
                                                              model_type='microsoft/deberta-xlarge-mnli', verbose=False)
                                    bert_f1_matrix[i, j] = F1[0].item()
                                except Exception:
                                    pass
                            if _BLEU_AVAILABLE:
                                try:
                                    bleu_matrix[i, j] = sentence_bleu([cq_man.split()], cq_gen.split(), smoothing_function=_smooth)
                                except Exception:
                                    pass
                            if self._rouge:
                                try:
                                    r = self._rouge.score(cq_man, cq_gen)["rougeLsum"]
                                    rougeL_matrix[i, j] = float(r.fmeasure)
                                except Exception:
                                    pass

                    def agg(mat):
                        return {'mean': float(np.mean(mat)), 'median': float(np.median(mat)),
                                'std': float(np.std(mat)), 'max': float(np.max(mat))}

                    metrics['pairwise_BERTScore_F1'] = agg(bert_f1_matrix)
                    metrics['pairwise_BLEU'] = agg(bleu_matrix)
                    metrics['pairwise_ROUGE_L_F1'] = agg(rougeL_matrix)
                except Exception as e:
                    metrics['pairwise_error'] = f"pairwise metrics failed: {e}"

            metrics['ontologies_found'] = ontologies
            metrics['pdfs_found'] = pdfs
            metrics['downloaded_ontologies'] = downloaded
            projects_out[str(key)] = metrics
            overall_rows.append(metrics)

        overall = {}
        numeric_keys = ['mean_cosine_all_pairs', 'per_generated_best_mean', 'per_gold_best_mean',
                        'precision_at_thr_generated', 'precision_at_thr_gold']
        for k in numeric_keys:
            vals = [r[k] for r in overall_rows if r.get(k) is not None]
            overall[f'{k}_mean'] = float(np.mean(vals)) if vals else None
            overall[f'{k}_median'] = float(np.median(vals)) if vals else None

        return {'by_group': projects_out, 'overall': overall, 'download_warnings': self._download_warnings}

    def validate(self, gold_question: str, generated_question: str) -> dict:
        input_text = f"Gold standard: {gold_question}\nGenerated: {generated_question}"

        try:
            missing_gold = pd.isna(gold_question)
        except Exception:
            missing_gold = gold_question is None
        try:
            missing_generated = pd.isna(generated_question)
        except Exception:
            missing_generated = generated_question is None

        if missing_gold or (isinstance(gold_question, str) and str(gold_question).strip() == ""):
            raise ValueError("Gold standard is missing or empty for this row.")
        if missing_generated or (isinstance(generated_question, str) and str(generated_question).strip() == ""):
            raise ValueError("Generated question is missing or empty for this row.")

        gold_question = str(gold_question)
        generated_question = str(generated_question)

        cq_manual = [q.strip() + "?" for q in gold_question.split("?") if q.strip()]
        cq_generated = [q.strip() + "?" for q in generated_question.split("?") if q.strip()]

        if not cq_manual or not cq_generated:
            raise ValueError("Both gold standard and generated questions must contain valid questions.")

        # SBERT cosine similarity
        embeddings = self._encode_with_cache(cq_generated + cq_manual)
        cosine_sim_matrix = cosine_similarity(
            embeddings[:len(cq_generated)], embeddings[len(cq_generated):]
        )

        # Jaccard
        def jaccard_similarity(str1, str2):
            set1, set2 = set(str1.split()), set(str2.split())
            inter = len(set1 & set2)
            union = len(set1 | set2)
            return inter / union if union else 0.0

        jaccard_sim_matrix = np.zeros((len(cq_generated), len(cq_manual)))
        for i, cq_gen in enumerate(cq_generated):
            for j, cq_man in enumerate(cq_manual):
                jaccard_sim_matrix[i, j] = jaccard_similarity(cq_gen, cq_man)

        # BERTScore
        bertscore_matrix = np.zeros((len(cq_generated), len(cq_manual)))
        precision_matrix = np.zeros((len(cq_generated), len(cq_manual)))
        recall_matrix = np.zeros((len(cq_generated), len(cq_manual)))
        bleu_matrix = np.zeros((len(cq_generated), len(cq_manual)))
        rougeL_f1_matrix = np.zeros((len(cq_generated), len(cq_manual)))

        for i, cq_gen in enumerate(cq_generated):
            for j, cq_man in enumerate(cq_manual):
                if _BERTSCORE_AVAILABLE:
                    try:
                        P, R, F1 = bert_score_fn([cq_gen], [cq_man], lang='en',
                                                  model_type='microsoft/deberta-xlarge-mnli', verbose=False)
                        precision_matrix[i, j] = P[0].item()
                        recall_matrix[i, j] = R[0].item()
                        bertscore_matrix[i, j] = F1[0].item()
                    except Exception:
                        pass
                if _BLEU_AVAILABLE:
                    try:
                        bleu_matrix[i, j] = sentence_bleu([cq_man.split()], cq_gen.split(), smoothing_function=_smooth)
                    except Exception:
                        pass
                if self._rouge:
                    try:
                        r = self._rouge.score(cq_man, cq_gen)["rougeLsum"]
                        rougeL_f1_matrix[i, j] = float(r.fmeasure)
                    except Exception:
                        pass

        similarity_results = []
        for i, cq_gen in enumerate(cq_generated):
            for j, cq_man in enumerate(cq_manual):
                similarity_results.append({
                    "Generated CQ": cq_gen,
                    "Manual CQ": cq_man,
                    "Cosine Similarity": cosine_sim_matrix[i, j],
                    "Jaccard Similarity": jaccard_sim_matrix[i, j],
                    "BERTScore-F1": bertscore_matrix[i, j],
                    "BERTScore-Precision": precision_matrix[i, j],
                    "BERTScore-Recall": recall_matrix[i, j],
                    "BLEU": bleu_matrix[i, j],
                    "ROUGE-L-F1": rougeL_f1_matrix[i, j],
                })
        sim_results_df = pd.DataFrame(similarity_results)

        def nn(x):
            return None if (x is None or (isinstance(x, float) and math.isnan(x))) else x

        avg_cosine = nn(float(sim_results_df['Cosine Similarity'].mean()))
        max_cosine = nn(float(sim_results_df['Cosine Similarity'].max()))
        avg_jaccard = nn(float(sim_results_df['Jaccard Similarity'].mean()))
        avg_bertscore_f1 = nn(float(sim_results_df['BERTScore-F1'].mean()))
        max_bertscore_f1 = nn(float(sim_results_df['BERTScore-F1'].max()))
        avg_bertscore_precision = nn(float(sim_results_df['BERTScore-Precision'].mean()))
        avg_bertscore_recall = nn(float(sim_results_df['BERTScore-Recall'].mean()))
        avg_bleu = nn(float(sim_results_df['BLEU'].mean()))
        max_bleu = nn(float(sim_results_df['BLEU'].max()))
        avg_rougeL = nn(float(sim_results_df['ROUGE-L-F1'].mean()))
        max_rougeL = nn(float(sim_results_df['ROUGE-L-F1'].max()))

        best_vec = self._best_match_vector(cosine_sim_matrix)
        min_best = nn(float(best_vec.min())) if best_vec.size else None
        max_best = nn(float(best_vec.max())) if best_vec.size else None
        avg_best = nn(float(best_vec.mean())) if best_vec.size else None
        threshold = 0.6
        matches_at_thr = self._matches_at_threshold(best_vec, thr=threshold)
        precision_at_thr = self._precision_at_threshold(best_vec, thr=threshold)

        # LLM analysis prompt
        sorted_pairs = sim_results_df.sort_values(by='Cosine Similarity', ascending=False).head(5)
        fmt = lambda x: f"{x:.2f}" if x is not None else "N/A"
        prompt = "Analyze the two sets of Competency Questions (CQ) generated and manual.\n\n"
        prompt += f"Statistics:\n- Average cosine similarity: {fmt(avg_cosine)}\n"
        prompt += f"- Max cosine similarity: {fmt(max_cosine)}\n"
        prompt += f"- Average Jaccard: {fmt(avg_jaccard)}\n"
        prompt += f"- Average BERTScore-F1: {fmt(avg_bertscore_f1)}\n"
        prompt += f"- Max BERTScore-F1: {fmt(max_bertscore_f1)}\n"
        prompt += f"- Average BLEU: {fmt(avg_bleu)}\n"
        prompt += f"- Average ROUGE-L F1: {fmt(avg_rougeL)}\n"
        prompt += f"- Best-match cosine per generated CQ — min: {fmt(min_best)}, max: {fmt(max_best)}, avg: {fmt(avg_best)}\n"
        prompt += f"- Precision@{threshold}: {precision_at_thr:.2f} ({matches_at_thr} matches)\n\n"
        prompt += "Pairs with highest similarity:\n"
        for _, row in sorted_pairs.iterrows():
            prompt += (f"- Generated: \"{row['Generated CQ']}\"  |  Manual: \"{row['Manual CQ']}\" "
                       f"(Cosine: {row['Cosine Similarity']:.2f}, BERTScore-F1: {row['BERTScore-F1']:.2f})\n")
        prompt += ("\nAnswer the following:\n"
                   "1. Which pairs have the highest similarity?\n"
                   "2. Which competency questions are missing and should be integrated? "
                   "Rank them by relevance (most relevant first). "
                   "Do not suggest questions already covered by existing ones. "
                   "Only suggest genuinely new questions not yet considered by the experts.\n"
                   "Answer clearly and in detail.")

        clean_analysis = None
        if self.validation_mode in {"llm", "all", "cosine_bertscore_judge"}:
            messages = [
                {"role": "system", "content": "You are a semantics expert assistant."},
                {"role": "user", "content": prompt},
            ]
            analysis = self.generate_response(chosen_model=self.model, messages=messages)
            clean_analysis = self.remove_html_tags(analysis)

        result = {}
        if self.validation_mode == "cosine_bertscore_judge":
            judge_scores = self.llm_judge_scores(cq_generated)
            result["Average Cosine Similarity"] = avg_cosine
            result["Max Cosine Similarity"] = max_cosine
            result["Average BERTScore-F1"] = avg_bertscore_f1
            result["Max BERTScore-F1"] = max_bertscore_f1
            result["Best-match Cosines"] = best_vec.tolist()
            result["Matches@0.6"] = matches_at_thr
            result["Precision@0.6"] = precision_at_thr
            result["LLM Analysis"] = clean_analysis
            result["LLM_as_Judge"] = [
                {"Relevance": int(r["Relevance"]), "Clarity": int(r["Clarity"]),
                 "Depth": int(r["Depth"]), "Average": float(r["Average"])}
                for r in judge_scores.to_dict(orient="records")
            ]
        elif self.validation_mode == "llm":
            result["LLM Analysis"] = clean_analysis
        elif self.validation_mode == "cosine":
            cosine_heatmap_base64 = generate_heatmap(cosine_sim_matrix, title="Cosine Similarity Heatmap")
            file_path = ""
            if self.output_folder:
                filename = f"cosine_heatmap_{abs(hash(input_text))}.png"
                file_path = save_heatmap_image(cosine_heatmap_base64, self.output_folder, filename)
            result["Average Cosine Similarity"] = avg_cosine
            result["Max Cosine Similarity"] = max_cosine
            result["Cosine Heatmap"] = file_path if file_path else "N/A"
            result["Best-match Cosines"] = best_vec.tolist()
            result["Matches@0.6"] = matches_at_thr
            result["Precision@0.6"] = precision_at_thr
        elif self.validation_mode == "jaccard":
            jaccard_heatmap_base64 = generate_heatmap(jaccard_sim_matrix, title="Jaccard Similarity Heatmap")
            file_path = ""
            if self.output_folder:
                filename = f"jaccard_heatmap_{abs(hash(input_text))}.png"
                file_path = save_heatmap_image(jaccard_heatmap_base64, self.output_folder, filename)
            result["Average Jaccard Similarity"] = avg_jaccard
            result["Jaccard Heatmap"] = file_path if file_path else "N/A"
        elif self.validation_mode == "bertscore":
            result["Average BERTScore-F1"] = avg_bertscore_f1
            result["Max BERTScore-F1"] = max_bertscore_f1
            result["Average BERTScore-Precision"] = avg_bertscore_precision
            result["Average BERTScore-Recall"] = avg_bertscore_recall
        else:  # "all"
            cosine_heatmap_base64 = generate_heatmap(cosine_sim_matrix, title="Cosine Similarity Heatmap")
            jaccard_heatmap_base64 = generate_heatmap(jaccard_sim_matrix, title="Jaccard Similarity Heatmap")
            file_path_cosine = file_path_jaccard = ""
            if self.output_folder:
                file_path_cosine = save_heatmap_image(cosine_heatmap_base64, self.output_folder,
                                                       f"cosine_heatmap_{abs(hash(input_text))}.png")
                file_path_jaccard = save_heatmap_image(jaccard_heatmap_base64, self.output_folder,
                                                        f"jaccard_heatmap_{abs(hash(input_text))}.png")
            result["LLM Analysis"] = clean_analysis
            result["Average Cosine Similarity"] = avg_cosine
            result["Max Cosine Similarity"] = max_cosine
            result["Average Jaccard Similarity"] = avg_jaccard
            result["Average BERTScore-F1"] = avg_bertscore_f1
            result["Max BERTScore-F1"] = max_bertscore_f1
            result["Average BERTScore-Precision"] = avg_bertscore_precision
            result["Average BERTScore-Recall"] = avg_bertscore_recall
            result["Average BLEU"] = avg_bleu
            result["Max BLEU"] = max_bleu
            result["Average ROUGE-L F1"] = avg_rougeL
            result["Max ROUGE-L F1"] = max_rougeL
            result["Cosine Heatmap"] = file_path_cosine if file_path_cosine else "N/A"
            result["Jaccard Heatmap"] = file_path_jaccard if file_path_jaccard else "N/A"
            result["Best-match Cosines"] = best_vec.tolist()
            result["Matches@0.6"] = matches_at_thr
            result["Precision@0.6"] = precision_at_thr

        return result

    def llm_judge_scores(self, questions: list) -> pd.DataFrame:
        instructions = (
            "Rate each question from 1 to 5 on three criteria: relevance, clarity, depth.\n"
            "Return one line per item as: <idx>. R=<1-5> C=<1-5> D=<1-5>\n"
            "No explanations."
        )
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        messages = [
            {"role": "system", "content": "You are a careful evaluator."},
            {"role": "user", "content": instructions + "\n\nQuestions:\n" + numbered},
        ]
        txt = self.generate_response(chosen_model=self.model, messages=messages, max_tokens=400, temperature=0)
        rows = []
        for line in txt.splitlines():
            m = re.search(r"^\s*(\d+)\.\s*R=(\d)\s*C=(\d)\s*D=(\d)\s*$", line.strip())
            if not m:
                continue
            idx, r, c, d = map(int, m.groups())
            rows.append({"idx": idx - 1, "Relevance": r, "Clarity": c, "Depth": d})
        if not rows:
            return pd.DataFrame(columns=["Relevance", "Clarity", "Depth", "Average"])
        df = pd.DataFrame(rows).sort_values("idx")
        df["Average"] = df[["Relevance", "Clarity", "Depth"]].mean(axis=1)
        return df.drop(columns=["idx"])
