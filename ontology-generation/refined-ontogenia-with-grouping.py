import pandas as pd
from openai import OpenAI
import logging
from datetime import datetime
import json

client = OpenAI(api_key='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')

def read_cqs_from_csv(file_path):
    try:
        data = pd.read_csv(file_path, quotechar='"', quoting=1)
        return data[['CQID', 'CQ', 'scenario']]
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame(columns=['CQID', 'CQ', 'scenario'])

procedure = "Define an ontology using an overall procedure based on understanding each competency question using this procedure: 1. Question understanding. 2. Preliminary identification of the context. 3. Critically assess the preliminary analysis. If not sure, try to reassess. 4. Confirm the final answer and explain the reasoning. 5. Confidence evaluation and explanation. This ontology has to be defined in OWL. Remember to add domain and range, and restrictions when needed. It must be complete and perfectly usable."

ontology_elements = "Classes, Object Properties, Datatype Properties. Object properties need to have domain and range. All of them need to have an explanation in the rdfs:label. You also need to add restrictions, and subclasses for both classes and object properties when applicable."

data = pd.read_csv('data/patterns.csv')  # Update the path to your CSV file
patterns_json = json.dumps({row['Name']: row['Pattern_owl'] for _, row in data.iterrows()})

def design_ontology(patterns_json, CQ, scenario, procedure, ontology_elements, previous_output=""):
    prompt = (
        f"Following the previous output: '{previous_output}' Read the following instructions: '{procedure}'. Based on the scenario: '{scenario}', design an ontology module that comprehensively answers the following competency question: '{CQ}'. You can use the following ontology design patterns in OWL format: {patterns_json}. Remember what are the ontology elements: {ontology_elements}. When you're done send me only the whole ontology you've designed in Turtle (.ttl) format, do not comment. "
    )

    messages = [
        {"role": "system", "content": "You are an ontology engineer expert."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="o1-preview",
        messages=messages,
        temperature=1,
        max_tokens=4096,
    )

    logging.info(f"Response at {datetime.now()}: {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

def process_sequential_cqs(cqs_data):
    scenario = cqs_data.iloc[0]['scenario']  # Use the scenario of the first CQ for all in the group
    cumulative_output = ""  # Initialize cumulative output for the entire sequence
    for index, row in cqs_data.iterrows():
        cq_id = row['CQID']
        cq_text = row['CQ']
        ontology_output = design_ontology(patterns_json, cq_text, scenario, procedure, ontology_elements, cumulative_output)
        cumulative_output += "\n" + ontology_output  # Append the current output for use in the next iteration
        print(f"Processed ontology for CQID {cq_id}: {cq_text}")
    return cumulative_output

def main():
    file_path = 'data/benchmarklimited.csv'
    cqs_data = read_cqs_from_csv(file_path)  # Ensure the file path is correct
    print("Select CQs from the following:")
    print(cqs_data[['CQID', 'CQ']])
    start = int(input("Start index: "))
    end = int(input("End index: "))
    selected_cqs = cqs_data[start:end+1]

    print("Selected CQs:")
    for index, row in selected_cqs.iterrows():
        print(f"{row['CQID']}: {row['CQ']}")
    confirmation = input("Proceed with these CQs? (yes/no): ")
    if confirmation.lower() != 'yes':
        print("Operation cancelled.")
        return

    final_ontology = process_sequential_cqs(selected_cqs)
    first_cqid = selected_cqs.iloc[0]['CQID']
    output_path = f'combined_ontology_output_reproducibility{first_cqid}.ttl'
    with open(output_path, 'w') as file:
        file.write(final_ontology)
    print(f"Final ontology output has been written to: {output_path}")

if __name__ == "__main__":
    main()
