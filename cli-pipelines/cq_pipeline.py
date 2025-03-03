#!/usr/bin/env python
# Example usage:
# python cq_pipeline.py --dataset_path "path/to/your.csv" --api_key YOUR_API_KEY --generate --validate --dataset_description "Your dataset description" --validation_mode all --output_folder my_heatmaps

import argparse
import os
import pandas as pd
import json
import openai
from cq_verification import validate_cq


def generate_cq(dataset_path, dataset_description):
    """
    Reads the CSV, extracts a dataset sample (first 10 rows) and an optional common scenario,
    then builds a prompt using defined competency question patterns and instructions.
    Calls the OpenAI API and returns the generated competency questions.
    """
    # Read CSV and extract a sample
    df = pd.read_csv(dataset_path)
    snippet_df = df.head(10)
    common_scenario = None
    if "scenario" in snippet_df.columns:
        scenarios = snippet_df["scenario"].unique()
        if len(scenarios) == 1:
            common_scenario = scenarios[0]
            snippet_df = snippet_df.drop("scenario", axis=1)
    dataset_sample = snippet_df.to_string(index=False)
    if common_scenario:
        dataset_sample += f"\nScenario: {common_scenario}"

    # Define patterns and instructions for generating competency questions
    patterns = [
        {"pattern": "Which [class expression 1][object property expression][class expression 2]?",
         "example": "Which pizzas contain pork?"},
        {"pattern": "How much does [class expression][datatype property]?",
         "example": "How much does Margherita Pizza weigh?"},
        {"pattern": "What type of [class expression] is [individual]?",
         "example": "What type of software (API, Desktop application etc.) is it?"},
        {"pattern": "Is the [class expression 1][class expression 2]?",
         "example": "Is the software open source development?"},
        {"pattern": "What [class expression] has the [numeric modifier][datatype property]?",
         "example": "What pizza has the lowest price?"},
        {"pattern": "Which are [class expressions]?",
         "example": "Which are gluten-free bases?"}
    ]
    instructions = [
        {
            "instruction": "Do not make explicit references to the dataset or its variables in the generated competency questions.",
            "example": {
                "incorrect": "How many cases of Salmonella were reported in Lombardy in 2020?",
                "correct": "How many cases of the disease were reported in the region in a given year?"
            }
        },
        {
            "instruction": "Keep the questions simple. Each competency question should not contain another simpler competency question within it.",
            "example": {
                "incorrect": "Who wrote The Hobbit and in what year was the book written?",
                "correct": ["Who wrote the book?", "In what year was the book written?"]
            }
        },
        {
            "instruction": "Do not include real entities; instead, abstract them into more generic concepts.",
            "example": {
                "incorrect": "Who is the author of 'Harry Potter'?",
                "correct": "Who is the author of the book?"
            }
        }
    ]
    clustering_instructions = (
        "Once the competency questions have been generated, they should be clustered into thematic areas. "
        "Each cluster represents an ontological module in the format: area : competency question and separated by ; . "
        "For example: Doctoral Theses Analysis : Which departments had new enrollments in a specific year?; "
        "Doctoral Theses Analysis : How many unique departments are listed in the dataset?;"
    )

    messages = [
        {"role": "system",
         "content": (
             "You are an ontology engineer. Generate a list of competency questions based on the dataset provided, "
             "following these patterns and instructions. Use the following competency question patterns:\n"
             f"{json.dumps(patterns)}\n\n"
             "Follow these instructions when generating the competency questions:\n"
             f"{json.dumps(instructions, indent=2)}\n\n"
             "After generating the questions, cluster them into thematic areas according to these guidelines:\n"
             f"{clustering_instructions}"
         )},
        {"role": "user",
         "content": f"Dataset description: {dataset_description}\n\nDataset sample: {dataset_sample}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=4000,
        temperature=0
    )
    generated_cq = response.choices[0].message.content.strip()
    return generated_cq


def main():
    parser = argparse.ArgumentParser(description="CQ Generation and Validation Pipeline")
    parser.add_argument("--dataset_path", required=True, help="Path to the CSV dataset")
    parser.add_argument("--api_key", required=False,
                        help="OpenAI API key (if not provided, the default in cq_verification.py is used)")
    parser.add_argument("--validation_mode", choices=["all", "cosine", "jaccard", "llm"], default="all",
                        help="Mode for CQ validation: 'all', 'cosine', 'jaccard', or 'llm'")
    parser.add_argument("--output_folder", default="heatmaps",
                        help="Folder in which to save heatmap images (if applicable)")
    parser.add_argument("--dataset_description", default="", help="Optional dataset description for CQ generation")
    parser.add_argument("--generate", action="store_true", help="Generate competency questions and update the CSV")
    parser.add_argument("--validate", action="store_true", help="Perform competency question validation")
    args = parser.parse_args()

    # Set the API key if provided
    if args.api_key:
        openai.api_key = args.api_key

    # Load CSV dataset
    try:
        df = pd.read_csv(args.dataset_path)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return

    # --- Generation Step ---
    # If --generate is specified or the CSV lacks a "generated" column, generate CQs.
    if args.generate or "generated" not in df.columns:
        print("Generating competency questions...")
        try:
            generated_cq = generate_cq(args.dataset_path, args.dataset_description)
            df["generated"] = generated_cq
            # Save the updated CSV to a new file (e.g., original filename_with_generated.csv)
            base, ext = os.path.splitext(args.dataset_path)
            updated_csv_path = base + "_with_generated" + ext
            df.to_csv(updated_csv_path, index=False)
            print(f"Generated competency questions added to CSV file: {updated_csv_path}")
        except Exception as e:
            print(f"Error during competency question generation: {str(e)}")
            return
    else:
        print("Skipping generation; 'generated' column already exists and --generate not specified.")

    # --- Validation Step ---
    if args.validate:
        print("Performing competency question validation...")
        # Ensure the CSV has the required "gold standard" column
        if "gold standard" not in df.columns:
            print("Error: CSV file must contain a 'gold standard' column for validation.")
            return
        results = []
        # For each row, create a prompt with the gold standard and generated CQs and validate them.
        for idx, row in df.iterrows():
            input_text = f"Gold standard: {row['gold standard']}\nGenerated: {row['generated']}"
            try:
                result_text = validate_cq(input_text, mode=args.validation_mode, output_folder=args.output_folder)
                results.append(f"Row {idx}:\n{result_text}\n")
            except Exception as e:
                results.append(f"Row {idx}:\nError: {str(e)}\n")
        output_text = ("\n" + "-" * 40 + "\n").join(results)
        validation_output_file = "cq_validation_results.txt"
        with open(validation_output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Validation results saved to {validation_output_file}")


if __name__ == "__main__":
    main()
