import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

from app.services.heatmap_generator import generate_heatmap, save_heatmap_image
from app.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class CQValidator:
    def __init__(self, output_folder: str, model: str = "gpt-4", validation_mode: str = "all"):
        self.output_folder = output_folder
        self.model = model
        self.validation_mode = validation_mode

    @staticmethod
    def remove_html_tags(text: str) -> str:
        return re.sub(r'<[^>]+>', '', text)

    def validate(self, gold_question: str, generated_question: str) -> dict:
        # Prepare input text for GPT and heatmap naming
        input_text = f"Gold standard: {gold_question}\nGenerated: {generated_question}"

        # Split the questions (ensuring each ends with a '?')
        cq_manual = [q.strip() + "?" for q in gold_question.split("?") if q.strip()]
        cq_generated = [q.strip() + "?" for q in generated_question.split("?") if q.strip()]

        if not cq_manual or not cq_generated:
            raise ValueError("Both gold standard and generated questions must contain valid questions.")

        # Calculate cosine similarity
        vectorizer = CountVectorizer().fit_transform(cq_generated + cq_manual)
        cosine_sim_matrix = cosine_similarity(
            vectorizer[:len(cq_generated)], vectorizer[len(cq_generated):]
        )

        # Calculate Jaccard similarity
        def jaccard_similarity(str1, str2):
            set1 = set(str1.split())
            set2 = set(str2.split())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union != 0 else 0

        jaccard_sim_matrix = np.zeros((len(cq_generated), len(cq_manual)))
        for i, cq_gen in enumerate(cq_generated):
            for j, cq_man in enumerate(cq_manual):
                jaccard_sim_matrix[i, j] = jaccard_similarity(cq_gen, cq_man)

        # Build a list of similarity results for each pair
        similarity_results = []
        for i, cq_gen in enumerate(cq_generated):
            for j, cq_man in enumerate(cq_manual):
                similarity_results.append({
                    "Generated CQ": cq_gen,
                    "Manual CQ": cq_man,
                    "Cosine Similarity": cosine_sim_matrix[i, j],
                    "Jaccard Similarity": jaccard_sim_matrix[i, j]
                })
        sim_results_df = pd.DataFrame(similarity_results)

        import math

        avg_cosine = sim_results_df['Cosine Similarity'].mean()
        max_cosine = sim_results_df['Cosine Similarity'].max()
        avg_jaccard = sim_results_df['Jaccard Similarity'].mean()

        # Replace NaN with None
        if math.isnan(avg_cosine):
            avg_cosine = None
        if math.isnan(max_cosine):
            max_cosine = None
        if math.isnan(avg_jaccard):
            avg_jaccard = None

        sorted_pairs = sim_results_df.sort_values(by='Cosine Similarity', ascending=False)
        max_pairs = 5
        selected_pairs = sorted_pairs.head(max_pairs)
        avg_cosine_str = f"{avg_cosine:.2f}" if avg_cosine is not None else "N/A"
        max_cosine_str = f"{max_cosine:.2f}" if max_cosine is not None else "N/A"
        avg_jaccard_str = f"{avg_jaccard:.2f}" if avg_jaccard is not None else "N/A"

        # Build the GPT prompt for analysis
        prompt = "Analyze the two sets of Competency Questions (CQ) generated and manual.\n\n"
        prompt += f"Statistics:\n- Average cosine similarity: {avg_cosine_str}\n"
        prompt += f"- Maximum cosine similarity: {max_cosine_str}\n"
        prompt += f"- Average Jaccard similarity: {avg_jaccard_str}\n\n"

        prompt += "Pairs with highest similarity:\n"
        for _, row in selected_pairs.iterrows():
            prompt += (f"- Generated: \"{row['Generated CQ']}\"  |  Manual: \"{row['Manual CQ']}\" "
                       f"(Cosine: {row['Cosine Similarity']:.2f}, Jaccard: {row['Jaccard Similarity']:.2f})\n")
        prompt += ("\nAnswer the following questions:\n"
                   "1. Which pairs of CQs have the highest similarity?\n"
                   "2. Which essential and important CQs are missing from the manual CQ list?\n"
                   "Answer clearly and in detail.")

        messages = [
            {"role": "system", "content": "You are a semantics expert assistant."},
            {"role": "user", "content": prompt}
        ]

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=400,
            temperature=0
        )
        gpt_analysis = response.choices[0].message.content.strip()
        clean_analysis = self.remove_html_tags(gpt_analysis)

        result = {}
        if self.validation_mode == "llm":
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
        elif self.validation_mode == "jaccard":
            jaccard_heatmap_base64 = generate_heatmap(jaccard_sim_matrix, title="Jaccard Similarity Heatmap")
            file_path = ""
            if self.output_folder:
                filename = f"jaccard_heatmap_{abs(hash(input_text))}.png"
                file_path = save_heatmap_image(jaccard_heatmap_base64, self.output_folder, filename)
            result["Average Jaccard Similarity"] = avg_jaccard
            result["Jaccard Heatmap"] = file_path if file_path else "N/A"
        else:  # mode "all"
            cosine_heatmap_base64 = generate_heatmap(cosine_sim_matrix, title="Cosine Similarity Heatmap")
            jaccard_heatmap_base64 = generate_heatmap(jaccard_sim_matrix, title="Jaccard Similarity Heatmap")
            file_path_cosine = ""
            file_path_jaccard = ""
            if self.output_folder:
                filename_cosine = f"cosine_heatmap_{abs(hash(input_text))}.png"
                file_path_cosine = save_heatmap_image(cosine_heatmap_base64, self.output_folder, filename_cosine)
                filename_jaccard = f"jaccard_heatmap_{abs(hash(input_text))}.png"
                file_path_jaccard = save_heatmap_image(jaccard_heatmap_base64, self.output_folder, filename_jaccard)
            result["LLM Analysis"] = clean_analysis
            result["Average Cosine Similarity"] = avg_cosine
            result["Max Cosine Similarity"] = max_cosine
            result["Average Jaccard Similarity"] = avg_jaccard
            result["Cosine Heatmap"] = file_path_cosine if file_path_cosine else "N/A"
            result["Jaccard Heatmap"] = file_path_jaccard if file_path_jaccard else "N/A"
        return result
