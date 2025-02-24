from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from openai import OpenAI
from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import uuid
import json

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app)

# Set your OpenAI API key
client = OpenAI(api_key='key')

# Parser for input arguments
parser = reqparse.RequestParser()
parser.add_argument('message', type=str, required=True, help="Message is required.")
app.config['UPLOAD_FOLDER'] = 'uploads'  # or some folder you prefer

# Make sure the folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    model_selected = request.form.get('model_select', 'gpt-3.5')

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Read the CSV
    try:
        df = pd.read_csv(file_path)
        # Generate a preview (first 5 rows) as string
        df_preview = df.head().to_string(index=False)

    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {str(e)}"}), 400

    # Check if "scenario" column is present (optional)
    if "scenario" not in df.columns:
        return jsonify({
            "message": "Upload successful, but 'scenario' not found. Please proceed to User Story Generation.",
            "model_selected": model_selected,
            "file_path": file_path
        }), 200

    # If scenario is present:
    return jsonify({
        "message": "Upload successful. 'scenario' column found.",
        "model_selected": model_selected,
        "file_path": file_path,
        "preview": df_preview
    }), 200


class GPTResourceUserStory(Resource):
    def post(self):
        local_parser = reqparse.RequestParser()
        local_parser.add_argument('message', type=str, required=True, help="Message is required.")
        local_parser.add_argument('file_path', type=str, required=False, default=None)
        args = local_parser.parse_args()
        user_message = args['message']
        file_path = args['file_path']

        # Step 1: Prepare partial dataset text (first 10 rows)
        dataset_snippet = ""
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Take the first 10 rows (or fewer if the file is shorter)
                snippet_df = df.head(10)
                dataset_snippet = snippet_df.to_string(index=False)
                print(dataset_snippet)
            except Exception as e:
                dataset_snippet = "Could not read the dataset. " + str(e)
        else:
            dataset_snippet = "No valid dataset provided."

        # Step 2: Build the prompt
        prompt = (
            f"Create a scenario based on the user's instructions.\n\n"
            f"User Instructions: {user_message}\n\n"
            f"Dataset (first 10 rows):\n{dataset_snippet}\n\n"
            "Be concise."
        )

        # Step 3: Call GPT
        try:
            messages = [
                {"role": "system", "content": "You are an ontology engineer expert."},
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=200,
            )

            response_text = response.choices[0].message.content.strip()

            # Step 4: Optionally store the scenario in the dataset
            updated_file_path = None
            if file_path and os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df['scenario'] = response_text  # set the same scenario for all rows, or modify logic as needed

                    # Save to a new file, or overwrite. Here we'll create a new file with a unique name:
                    base, ext = os.path.splitext(os.path.basename(file_path))
                    new_filename = f"{base}_with_scenario_{uuid.uuid4().hex[:6]}{ext}"
                    updated_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                    df.to_csv(updated_file_path, index=False)
                except Exception as e:
                    return {'error': f"Could not update dataset: {str(e)}"}, 500

            # Step 5: Return scenario and download link (if we have an updated file)
            download_url = None
            if updated_file_path:
                # We'll serve downloads from /download/<filename>, so we only return the filename here
                filename_only = os.path.basename(updated_file_path)
                download_url = f"/download-updated/{filename_only}"

            return {
                       'response': response_text,
                       'download_url': download_url,
                       'message': "User story generated. You can now proceed to CQ Generation."
                   }, 200

        except Exception as e:
            return {'error': str(e)}, 500


# CQ Generation
class CQGeneration(Resource):
    def post(self):
        local_parser = reqparse.RequestParser()
        local_parser.add_argument('file_path', type=str, required=True, help="File path is required.")
        local_parser.add_argument('dataset_description', type=str, required=False, default="")
        local_parser.add_argument('message', type=str, required=False, default="")
        args = local_parser.parse_args()

        file_path = args['file_path']
        dataset_description = args['dataset_description'] or ""
        user_instructions = args['message']

        # If user instructions are provided and not already in dataset_description, append them.
        # Append user instructions if not already included
        if user_instructions and user_instructions.strip().lower() not in dataset_description.strip().lower():
            dataset_description += "\nUser Instructions: " + user_instructions.strip()

        try:
            if not os.path.exists(file_path):
                return {'error': f"File at {file_path} not found."}, 400

            # Read the CSV from the provided file_path
            df = pd.read_csv(file_path)

            # Get the first 5 rows as a preview
            preview_df = df.head(5)
            print(preview_df)

            common_scenario = None
            if "scenario" in preview_df.columns:
                scenario_values = preview_df["scenario"].unique()
                if len(scenario_values) == 1:
                    common_scenario = scenario_values[0]
                    preview_df = preview_df.drop("scenario", axis=1)

            # Convert the preview dataframe to string without index
            dataset_sample = preview_df.to_string(index=False)

            # Append the common scenario only once if it exists
            if common_scenario:
                dataset_sample += f"\nScenario: {common_scenario}"
                print(dataset_sample)

            # Define patterns for generating competency questions
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
                {"pattern": "Which are [class expressions]?", "example": "Which are gluten-free bases?"},
            ]

            # Define instructions for generating competency questions
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

            # Define instructions for clustering the competency questions into thematic areas
            clustering_instructions = "Once the competency questions have been generated, they should be clustered into thematic areas. Each cluster represents an ontological module in the format: area : competency question and separated by ; . For example: Doctoral Theses Analysis : Which departments had new enrollments in a specific year?; Doctoral theses Analysis : How many unique departments are listed in the dataset?;"

            # Define the messages sent to OpenAI for CQ generation
            messages = [
                {"role": "system",
                 "content": f"You are an ontology engineer. Generate a list of competency questions based on the dataset provided, following these patterns and instructions.Use the following competency question patterns:\n{json.dumps(patterns)} \n\n Follow these instructions when generating the competency questions:\n{json.dumps(instructions, indent=2)} \n\n After generating the questions, cluster them into thematic areas according to these guidelines:\n{clustering_instructions}."},
                {"role": "user",
                 "content": f"Dataset description: {dataset_description}\n\nDataset sample: {dataset_sample}"},

            ]
            print(messages)

            # Make the API call to OpenAI to get a response for the competency questions
            response = client.chat.completions.create(
                model="gpt-3.5",
                messages=messages,
                max_tokens=4000,
                temperature=0
            )

            response = response.choices[0].message.content.strip()
            print("llm response: ", response)

            return {'response': response}, 200
        except Exception as e:
            return {'error': str(e)}, 500

#cqval
class CQValidation(Resource):
    def post(self):
        args = parser.parse_args()
        user_message = args['message']
        try:
            prompt = f"Validate the competency questions provided: {user_message}"
            messages = [
                {"role": "system", "content": "You are an ontology engineer expert."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            return {'response': response_text}, 200
        except Exception as e:
            return {'error': str(e)}, 500


# Ontology Generation
class OntologyGeneration(Resource):
    def post(self):
        args = parser.parse_args()
        user_message = args['message']
        try:
            prompt = f"Generate an ontology for: {user_message}"
            messages = [
                {"role": "system", "content": "You are an ontology engineer expert."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            return {'response': response_text}, 200
        except Exception as e:
            return {'error': str(e)}, 500


# Ontology Testing
class OntologyTesting(Resource):
    def post(self):
        args = parser.parse_args()
        user_message = args['message']
        try:
            prompt = f"Test the following ontology: {user_message}"
            messages = [
                {"role": "system", "content": "You are an ontology engineer expert."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            return {'response': response_text}, 200
        except Exception as e:
            return {'error': str(e)}, 500


@app.route('/')
def index():
    return render_template('index.html')


# Add the resource to the API
# api.add_resource(GPTResourceUserStory, '/process')
api.add_resource(GPTResourceUserStory, '/user-story')
api.add_resource(CQGeneration, '/cq-generation')
api.add_resource(CQValidation, '/cq-validation')
api.add_resource(OntologyGeneration, '/ontology-generation')
api.add_resource(OntologyTesting, '/ontology-testing')
if __name__ == '__main__':
    app.run(debug=True)
