#python cq-verification-from-cli.py --C:..csv --mode all --output_folder my_heatmaps --api_key sk-mykey


import argparse
import os
import pandas as pd
from cq_verification import validate_cq

def main():
    parser = argparse.ArgumentParser(description="CQ Verification Tool")
    parser.add_argument("--dataset_path", required=True, help="Path to the CSV dataset")
    parser.add_argument("--api_key", required=False, help="OpenAI API key (if not provided, will use default in cq_verification.py)")
    parser.add_argument("--mode", choices=["all", "cosine", "jaccard", "llm"], default="all",
                        help="Mode of operation: 'all' returns full analysis (LLM comment + both heatmaps), 'llm' returns only the GPT analysis, 'cosine' or 'jaccard' return only that metric and save the respective heatmap")
    parser.add_argument("--output_folder", default="heatmaps", help="Folder in which to save heatmap images (if applicable)")
    args = parser.parse_args()

    # If an API key is provided, override the default key in cq_verification.py
    if args.api_key:
        import openai
        openai.api_key = args.api_key

    # Load the CSV dataset
    df = pd.read_csv(args.dataset_path)
    # Check that the required columns exist
    if "gold standard" not in df.columns or "generated" not in df.columns:
        print("Error: CSV file must contain 'gold standard' and 'generated' columns.")
        return

    # Process each row and collect results in plain text format
    results = []
    for idx, row in df.iterrows():
        input_text = f"Gold standard: {row['gold standard']}\nGenerated: {row['generated']}"
        try:
            result_text = validate_cq(input_text, mode=args.mode, output_folder=args.output_folder)
            results.append(f"Row {idx}:\n{result_text}\n")
        except Exception as e:
            results.append(f"Row {idx}:\nError: {str(e)}\n")

    # Combine results into one plain text output (separating each row's result)
    output_text = ("\n" + "-"*40 + "\n").join(results)

    output_file = "cq_verification_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

