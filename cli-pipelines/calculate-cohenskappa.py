import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from itertools import combinations


def calculate_cohens_kappa_from_csv(csv_file, annotator_sets):
    """
    Calculate Cohen's Kappa agreement score between multiple annotators,
    ignoring 'N/A' values and handling data type issues, from a CSV file.
    Computes the average kappa across all annotator sets and converts it into agreement percentage.
    Also calculates the percentage of cases where both source and target are considered "yes" by the majority on the same row.

    Parameters:
    csv_file (str): Path to the CSV file.
    annotator_sets (list of list): List containing lists of annotator column names.

    Returns:
    dict: Dictionary with Cohen's Kappa scores for each pair of annotators, overall average score, agreement percentage, and majority "yes" percentage on the same row.
    """
    # Load CSV file
    df = pd.read_csv(csv_file)

    all_kappa_scores = {}
    overall_kappa_values = []

    for annotators in annotator_sets:
        kappa_scores = {}

        # Ensure all annotator columns are treated as strings
        df[annotators] = df[annotators].astype(str)

        # Compute Cohen's Kappa for all annotator pairs
        for col1, col2 in combinations(annotators, 2):
            df_filtered = df[(df[col1] != 'N/A') & (df[col2] != 'N/A')]

            # Ensure categories are consistent
            unique_labels = list(set(df_filtered[col1].unique()) | set(df_filtered[col2].unique()))

            kappa = cohen_kappa_score(df_filtered[col1], df_filtered[col2], labels=unique_labels)
            kappa_scores[f"{col1} vs {col2}"] = kappa
            overall_kappa_values.append(kappa)

        all_kappa_scores.update(kappa_scores)

    # Compute the overall average kappa
    overall_avg_kappa = np.mean(overall_kappa_values)
    agreement_percentage = (overall_avg_kappa + 1) / 2 * 100  # Convert Kappa to percentage
    all_kappa_scores["Overall Average Kappa"] = overall_avg_kappa
    all_kappa_scores["Agreement Percentage"] = agreement_percentage

    # Compute majority "yes" agreement for both source and target on the same row
    source_cols, target_cols = annotator_sets
    df_filtered = df.dropna(subset=source_cols + target_cols)

    def majority_yes(row, cols):
        return (row[cols] == "yes").sum() > len(cols) / 2

    majority_yes_count = df_filtered.apply(
        lambda row: majority_yes(row, source_cols) and majority_yes(row, target_cols), axis=1).sum()
    majority_yes_percentage = (majority_yes_count / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0

    all_kappa_scores["Majority Yes Agreement Percentage (Same Row)"] = majority_yes_percentage

    return all_kappa_scores


# Example usage
csv_file = "wacgro-allevals - merged_wachgro_for_analysis_sandt.csv"  # Updated with actual file path
annotator_sets = [
    [
        "is the predicted target the same or very similar to the target domain?",
        "anno2is the predicted target the same or very similar to the target domain?",
        "anno3 is the predicted target the same or very similar to the target domain?"
    ],
    [
        "is the predicted source the same or very similar to the source domain?",
        "anno2is the predicted source the same or very similar to the source domain?",
        "anno3 is the predicted source the same or very similar to the source domain?"
    ]
]  # Updated with actual column names

kappa_scores = calculate_cohens_kappa_from_csv(csv_file, annotator_sets)
for pair, score in kappa_scores.items():
    print(f"Cohen's Kappa Score for {pair}: {score}")
