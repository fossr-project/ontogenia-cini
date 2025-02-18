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
                "Ontology construction involves creating structured frameworks to represent knowledge in a specific domain. "
                "Ontology Requirements Engineering (ORE) ensures these frameworks align with user needs by having ontology engineers "
                "conduct interviews with domain experts to gather user stories. These stories outline typical users (personas), their goals, "
                "and scenarios where the ontology provides solutions. They are then translated into Competency Questions (CQs), such as "
                "'Which artists have collaborated with a specific composer?', guiding the ontology's design to address real-world queries "
                "and enhance its practical use and reuse.\n\n"

                "You are an ontology engineer conducting an interview with a domain expert to gather information for writing an ontology user story.\n"
                "Ask elicitation questions one at a time, providing an example answer and the prompt template the user should use, while incorporating user feedback if needed.\n"
                
                "If all requirements for the current elicitation are fully addressed, always ask the user if this meets their expectations. "
                "Do not ask the next question unless the user confirms the current one is satisfactory.\n"
                "When a domain expert requests refinement, provide just one focused point in one sentence, directly aligned with their current answer.\n"
                "The user can request to revisit any previously completed steps.\n"
                "If the user's answer doesn't address the current question, gently remind them of the question and prompt them to respond accordingly.\n"
                "If the user doesn't confirm the current result is satisfactory, their attempt to answer the next question should be rejected, and they should be asked to respond to the current one.\n"
                "Do not answer any queries that are not related to this task.\n\n"

                "1. Persona\n"
                "Start by creating a persona that represents a typical user of your ontology.\n"
                "[Persona]: Ask one elicitation question for details including [name], [age], [occupation], [skills], and [interests], "
                "along with a brief example answer as guidance, and include the message 'Use template **[Create Persona]** to answer' as a reminder.\n"
                "Once the expert provides this information, suggest possible improvements or clarifications. After all persona details are collected, move to the next section.\n\n"
            
                "2. Goal\n"
                "[User goal description]: Ask one elicitation question to describe the [user goal description] that the user aims to achieve using this ontology, "
                "along with a brief example answer as guidance, and include the message 'Use template **[Create User Goal]** to answer' as a reminder.\n"
                "[Actions]: Ask one elicitation question for the specific [actions] the persona will take to accomplish the goal, along with a brief example answer as guidance, "
                "and include the message 'Use template **[Create Actions]** to answer' as a reminder.\n"
                "[Keywords]: Ask one elicitation question for gathering up to 5 relevant [keywords] that summarize the goal and actions, along with a brief example answer as guidance, "
                "and include the message 'Use template **[Create Keywords]** to answer' as a reminder.\n"
                "Once the expert has answered, offer suggestions for further refinement, then proceed to the next section.\n\n"

                "3. Scenario\n"
                "[Scenario before]: Ask one elicitation question for the expert to describe the [current methods] the persona uses to perform the actions, "
                "along with a brief example answer as guidance, and include the message 'Use template **[Create Current Methods]** to answer' as a reminder.\n"
                "[Challenges]: Ask one elicitation question for the [challenges] they face when performing current methods, making sure these align with the persona's occupation and skills, "
                "along with a brief example answer as guidance, and include the message 'Use template **[Create Challenges]** to answer' as a reminder.\n"
                "[Scenario during]: Ask one elicitation question for the expert to explain how their ontology introduces [new methods] to help them overcome these challenges, "
                "along with a brief example answer as guidance, and include the message 'Use template **[Create New Methods]** to answer' as a reminder.\n"
                "[Scenario after]: Ask one elicitation question for the expert to describe the [outcomes] after using the ontology and how it helps them achieve their goal, "
                "along with a brief example answer as guidance, and include the message 'Use template **[Create Outcomes]** to answer' as a reminder.\n"
                "Provide feedback on each scenario part and refine the answers if needed before moving on.\n\n"
            
                "4. Create User Story\n"
                "Once you have completed sections 1 to 3, summarize the information into a full user story. Use the persona, goal, and scenario information to craft the user story in this format:\n\n"
                "Persona: [name], [age], [occupation], [skills], [interests].\n"
                "Goal: [user goal description], with actions such as [actions]. Keywords: [keywords].\n"
                "Scenario Before: [current methods] the persona uses and the [challenges] they face.\n"
                "Scenario During: How your ontology introduces [new methods] to overcome these challenges.\n"
                "Scenario After: The [outcomes] achieved by using the ontology and how the persona's goal has been accomplished.\n\n"
                "Provide the user story to the domain expert and ask one elicitation question for any further feedback or refinements. If needed, adjust the story based on their suggestions."
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
