"""
CQ Comparison
"""

import openai
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score

import seaborn as sns
import matplotlib.pyplot as plt

# Configura la tua API key di OpenAI
openai.api_key = "my_key"

# Imposta il modello GPT da utilizzare
GPT_MODEL = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Funzione per ottenere embeddings da OpenAI
def get_openai_embeddings(cqs):
    """
    Ottieni embeddings da OpenAI per una lista di CQ.
    """
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=cqs
    )
    embeddings = [item['embedding'] for item in response['data']]
    return embeddings

# Funzione per calcolare le metriche di similarità
def calculate_similarity_metrics(cq_generated, cq_manual):
    """
    Calcola le metriche di similarità tra due liste di CQ.
    """
    # Ottieni embeddings da OpenAI
    embeddings_generated = np.array(get_openai_embeddings(cq_generated))
    embeddings_manual = np.array(get_openai_embeddings(cq_manual))
    
    # Similarità coseno
    cosine_sim = cosine_similarity(embeddings_generated, embeddings_manual)

    # Similarità Jaccard
    vectorizer = CountVectorizer(binary=True).fit(cq_generated + cq_manual)
    binary_generated = vectorizer.transform(cq_generated).toarray()
    binary_manual = vectorizer.transform(cq_manual).toarray()
    jaccard_sim = np.array([[jaccard_score(gen, man) for man in binary_manual] for gen in binary_generated])

    # Preparare il dataframe con i risultati
    results = []
    for i, gen in enumerate(cq_generated):
        for j, man in enumerate(cq_manual):
            results.append({
                'Generated CQ': gen,
                'Manual CQ': man,
                'Cosine Similarity': cosine_sim[i, j],
                'Jaccard Similarity': jaccard_sim[i, j],
            })

    results_df = pd.DataFrame(results)
    return results_df, cosine_sim, jaccard_sim

# Funzione per visualizzare la heatmap con legenda
def visualize_similarity_matrix_with_legend(
    similarity_matrix, labels_x, labels_y, title="Similarity Heatmap"
):
    """
    Visualizza una heatmap delle similarità con legenda.
    """
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=[f"CQ{i+1}" for i in range(len(labels_x))],
        yticklabels=[f"CQ{i+1}" for i in range(len(labels_y))],
        cmap="coolwarm",
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=18)
    plt.xlabel("Manual CQs", fontsize=14)
    plt.ylabel("Generated CQs", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.show()

# Funzione per ottenere un'analisi semantica approfondita dal modello GPT
def semantic_analysis_with_gpt(cq_generated, cq_manual, similarity_results, max_pairs=5):
    """
    Usa GPT per analizzare semanticamente i risultati delle similarità.
    Limita il numero di coppie da analizzare per evitare superamenti di token.
    """
    # Filtra solo le coppie con alta similarità (Cosine Similarity > 0.7)
    high_similarity = similarity_results[similarity_results['Cosine Similarity'] > 0.7]
    
    # Seleziona solo le prime 'max_pairs' coppie
    high_similarity = high_similarity.head(max_pairs)
    
    if high_similarity.empty:
        return "Non sono state trovate coppie di CQ con alta similarità."

    # Costruisci il prompt per GPT
    prompt = """
    Analizza i due set di Competency Questions (CQ) generati e manuali. 
    Rispondi alle seguenti domande:
    
    1. Perché le CQ elencate di seguito sono semanticamente simili?
    2. Come si potrebbero migliorare le CQ manuali per maggiore chiarezza semantica?
    3. Fornisci un'interpretazione dei risultati di similarità basati su cosine e Jaccard.
    
    Coppie di CQ simili:
    """
    
    for _, row in high_similarity.iterrows():
        prompt += f"- Generated CQ: \"{row['Generated CQ']}\"\n  Manual CQ: \"{row['Manual CQ']}\"\n"

    prompt += "\nRispondi in modo chiaro e dettagliato."

    # Richiesta al modello GPT
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[{"role": "system", "content": "Sei un assistente esperto di semantica."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Funzione principale
def main():
    """
    Workflow principale per l'analisi delle CQ.
    """
    print("=== Analisi delle Competency Questions (CQ) ===")
    print("\nInserisci le CQ generate automaticamente (separate da '?'):")
    cq_generated_input = input("> ").strip()
    cq_generated = [cq.strip() + "?" for cq in cq_generated_input.split("?") if cq.strip()]

    print("\nInserisci le CQ manuali (separate da '?'):")
    cq_manual_input = input("> ").strip()
    cq_manual = [cq.strip() + "?" for cq in cq_manual_input.split("?") if cq.strip()]

    # Calcola le metriche di similarità
    print("\nCalcolo delle metriche di similarità...")
    similarity_results, cosine_sim_matrix, jaccard_sim_matrix = calculate_similarity_metrics(cq_generated, cq_manual)

    # Visualizza i risultati
    print("\nTabella dei risultati:")
    print(similarity_results)

    print("\nVisualizzazione delle heatmap...")
    visualize_similarity_matrix_with_legend(
        cosine_sim_matrix, cq_manual, cq_generated, title="Cosine Similarity Heatmap"
    )

    # Analisi semantica con GPT
    print("\nAnalisi semantica con GPT...")
    gpt_analysis = semantic_analysis_with_gpt(cq_generated, cq_manual, similarity_results)
    print("\n=== Analisi GPT ===")
    print(gpt_analysis)

# Esegui il programma
main()

