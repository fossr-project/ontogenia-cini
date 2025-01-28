from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from openai import OpenAI
from flask import Flask, request, jsonify, render_template

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app)

# Set your OpenAI API key

# Parser for input arguments
parser = reqparse.RequestParser()
parser.add_argument('message', type=str, required=True, help="Message is required.")


class GPTResourceUserStory(Resource):
    def post(self):
        # Parse the incoming JSON data
        args = parser.parse_args()
        user_message = args['message']

        # Call GPT to process the message
        try:
            prompt = (
                f"Create a scenario basing on the user's instructions: {user_message} "
            )

            messages = [
                {"role": "system", "content": "You are an ontology engineer expert."},
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=100,
            )

            response_text = response.choices[0].message.content.strip()
            return {'response': response_text}, 200
        except Exception as e:
            return {'error': str(e)}, 500

# CQ Generation
class CQGeneration(Resource):
    def post(self):
        args = parser.parse_args()
        user_message = args['message']
        try:

            prompt = f"Generate competency questions for: {user_message}"
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

# CQ Validation
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
                max_tokens=150,
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
#api.add_resource(GPTResourceUserStory, '/process')
api.add_resource(GPTResourceUserStory, '/user-story')
api.add_resource(CQGeneration, '/cq-generation')
api.add_resource(CQValidation, '/cq-validation')
api.add_resource(OntologyGeneration, '/ontology-generation')
api.add_resource(OntologyTesting, '/ontology-testing')
if __name__ == '__main__':
    app.run(debug=True)
