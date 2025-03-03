import re
import io
import base64
import openai
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GPT_MODEL = "gpt-4"
openai.api_key = "key"

def generate_heatmap(similarity_matrix, title="Heatmap"):
    """
    Generate a heatmap using seaborn and return it as a base64-encoded PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def save_heatmap_image(encoded_image, output_folder, name):
    """
    Save the base64 encoded image to a file.
    """
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_data = base64.b64decode(encoded_image)
    file_path = os.path.join(output_folder, name)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_path

def validate_cq(input_text, mode="all", output_folder=None):
    """
    Validates competency questions (CQ) given input_text in the following format:

    Gold standard:
    cq1? cq2? cq3?

    Generated:
    cq1? cq2? cq3?

    Parameters:
      mode: one of "all", "cosine", "jaccard", "llm"
      output_folder: folder path to save heatmap images if applicable

    Returns:
      An HTML string containing the analysis and, if requested, the similarity metrics and saved heatmap file paths.
    """
    # Extract the two sections using regex
    gold_match = re.search(r"Gold\s*standard\s*:(.*?)Generated\s*:", input_text, re.IGNORECASE | re.DOTALL)
    if gold_match:
        gold_text = gold_match.group(1).strip()
    else:
        raise ValueError("Input must contain a 'Gold standard:' section followed by a 'Generated:' section.")

    generated_match = re.search(r"Generated\s*:(.*)", input_text, re.IGNORECASE | re.DOTALL)
    if generated_match:
        generated_text = generated_match.group(1).strip()
    else:
        raise ValueError("Input must contain a 'Generated:' section.")

    # Split questions by '?' and ensure each ends with '?'
    cq_manual = [q.strip() + "?" for q in gold_text.split("?") if q.strip()]
    cq_generated = [q.strip() + "?" for q in generated_text.split("?") if q.strip()]

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

    # Build a list of similarity results for each CQ pair
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

    # Compute overall statistics and select the top 5 pairs by cosine similarity
    avg_cosine = sim_results_df['Cosine Similarity'].mean()
    max_cosine = sim_results_df['Cosine Similarity'].max()
    avg_jaccard = sim_results_df['Jaccard Similarity'].mean()

    sorted_pairs = sim_results_df.sort_values(by='Cosine Similarity', ascending=False)
    max_pairs = 5
    selected_pairs = sorted_pairs.head(max_pairs)

    # Build the GPT prompt for analysis
    prompt = "Analizza i due set di Competency Questions (CQ) generati e manuali.\n\n"
    prompt += f"Statistiche:\n- Similarità coseno media: {avg_cosine:.2f}\n"
    prompt += f"- Similarità coseno massima: {max_cosine:.2f}\n"
    prompt += f"- Similarità Jaccard media: {avg_jaccard:.2f}\n\n"
    prompt += "Coppie con maggiore similarità:\n"
    for _, row in selected_pairs.iterrows():
        prompt += (f"- Generated: \"{row['Generated CQ']}\"  |  Manual: \"{row['Manual CQ']}\" "
                   f"(Cosine: {row['Cosine Similarity']:.2f}, Jaccard: {row['Jaccard Similarity']:.2f})\n")
    prompt += ("\nRispondi alle seguenti domande:\n"
               "1. Quali sono le coppie di CQ con maggiore similarità?\n"
               "2. Quali CQ essenziali e importanti mancano alla lista di CQ manuali?\n"
               "Rispondi in modo chiaro e dettagliato.")

    # Call GPT to get the analysis
    messages = [
        {"role": "system", "content": "Sei un assistente esperto di semantica."},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=400,
        temperature=0
    )
    gpt_analysis = response.choices[0].message.content.strip()

    # Construct output based on mode
    if mode == "llm":
        return f"<div><h2>LLM Analysis</h2><p>{gpt_analysis}</p></div>"
    elif mode == "cosine":
        cosine_heatmap_base64 = generate_heatmap(cosine_sim_matrix, "Cosine Similarity Heatmap")
        file_path = ""
        if output_folder:
            filename = f"cosine_heatmap_{abs(hash(input_text))}.png"
            file_path = save_heatmap_image(cosine_heatmap_base64, output_folder, filename)
        return f"<div><h2>Cosine Similarity Metrics</h2><p>Average cosine: {avg_cosine:.2f}, Max cosine: {max_cosine:.2f}</p><p>Cosine heatmap saved to: {file_path if file_path else 'N/A'}</p></div>"
    elif mode == "jaccard":
        jaccard_heatmap_base64 = generate_heatmap(jaccard_sim_matrix, "Jaccard Similarity Heatmap")
        file_path = ""
        if output_folder:
            filename = f"jaccard_heatmap_{abs(hash(input_text))}.png"
            file_path = save_heatmap_image(jaccard_heatmap_base64, output_folder, filename)
        return f"<div><h2>Jaccard Similarity Metrics</h2><p>Average jaccard: {avg_jaccard:.2f}</p><p>Jaccard heatmap saved to: {file_path if file_path else 'N/A'}</p></div>"
    else:  # mode == "all"
        cosine_heatmap_base64 = generate_heatmap(cosine_sim_matrix, "Cosine Similarity Heatmap")
        jaccard_heatmap_base64 = generate_heatmap(jaccard_sim_matrix, "Jaccard Similarity Heatmap")
        file_path_cosine = file_path_jaccard = ""
        if output_folder:
            filename_cosine = f"cosine_heatmap_{abs(hash(input_text))}.png"
            file_path_cosine = save_heatmap_image(cosine_heatmap_base64, output_folder, filename_cosine)
            filename_jaccard = f"jaccard_heatmap_{abs(hash(input_text))}.png"
            file_path_jaccard = save_heatmap_image(jaccard_heatmap_base64, output_folder, filename_jaccard)
        final_html = f"""
        <div style="font-family: monospace; white-space: pre-wrap;">
          <h2>LLM Analysis</h2>
          <p>{gpt_analysis}</p>
          <h2>Cosine Similarity Metrics</h2>
          <p>Average cosine: {avg_cosine:.2f}, Max cosine: {max_cosine:.2f}</p>
          <p>Cosine heatmap saved to: {file_path_cosine if file_path_cosine else 'N/A'}</p>
          <h2>Jaccard Similarity Metrics</h2>
          <p>Average jaccard: {avg_jaccard:.2f}</p>
          <p>Jaccard heatmap saved to: {file_path_jaccard if file_path_jaccard else 'N/A'}</p>
        </div>
        """
        return final_html
