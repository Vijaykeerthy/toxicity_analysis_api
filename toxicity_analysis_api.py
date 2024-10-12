from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flasgger import Swagger
import pickle
import numpy as np

app = Flask(__name__)
api = Api(app)
swagger = Swagger(app)

# Function to load pickled files safely
def load_pickle_file(file_path, file_type):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_type} from '{file_path}': {e}")
        return None

# Define the path to the model directory
model_dir = "model/"  # Ensure this folder is in the same directory as your script

# Load the TF-IDF vectorizers for each category
tox = load_pickle_file(f"{model_dir}toxic_vect.pkl", "toxic vectorizer")
sev = load_pickle_file(f"{model_dir}severe_toxic_vect.pkl", "severe toxic vectorizer")
obs = load_pickle_file(f"{model_dir}obscene_vect.pkl", "obscene vectorizer")
ins = load_pickle_file(f"{model_dir}insult_vect.pkl", "insult vectorizer")
thr = load_pickle_file(f"{model_dir}threat_vect.pkl", "threat vectorizer")
ide = load_pickle_file(f"{model_dir}identity_hate_vect.pkl", "identity hate vectorizer")

# Load the pickled Random Forest models for each category
tox_model = load_pickle_file(f"{model_dir}toxic_model.pkl", "toxic model")
sev_model = load_pickle_file(f"{model_dir}severe_toxic_model.pkl", "severe toxic model")
obs_model = load_pickle_file(f"{model_dir}obscene_model.pkl", "obscene model")
ins_model = load_pickle_file(f"{model_dir}insult_model.pkl", "insult model")
thr_model = load_pickle_file(f"{model_dir}threat_model.pkl", "threat model")
ide_model = load_pickle_file(f"{model_dir}identity_hate_model.pkl", "identity hate model")

class PredictToxicity(Resource):
    def get(self):
        """
        Predict the toxicity of the given text.
        ---
        tags:
          - Toxicity Prediction
        parameters:
          - name: text
            in: query
            type: string
            required: true
            description: The text to analyze for toxicity
        responses:
          200:
            description: Prediction results for the provided text
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    original_text:
                      type: string
                      description: The original input text
                    toxicity_results:
                      type: object
                      properties:
                        Prob (Toxic):
                          type: number
                          description: Probability of the text being toxic
                        Prob (Severe Toxic):
                          type: number
                          description: Probability of severe toxicity
                        Prob (Obscene):
                          type: number
                          description: Probability of the text being obscene
                        Prob (Insult):
                          type: number
                          description: Probability of the text being an insult
                        Prob (Threat):
                          type: number
                          description: Probability of the text containing threats
                        Prob (Identity Hate):
                          type: number
                          description: Probability of identity-based hate
          400:
            description: Input text is missing
        """
        text = request.args.get('text', '').strip()  # Safely handle empty input

        if not text:
            return jsonify({"error": "Text is required"}), 400

        data = [text]

        try:
            # Perform predictions for each category
            vect = tox.transform(data)
            pred_tox = tox_model.predict_proba(vect)[:, 1]

            vect = sev.transform(data)
            pred_sev = sev_model.predict_proba(vect)[:, 1]

            vect = obs.transform(data)
            pred_obs = obs_model.predict_proba(vect)[:, 1]

            vect = thr.transform(data)
            pred_thr = thr_model.predict_proba(vect)[:, 1]

            vect = ins.transform(data)
            pred_ins = ins_model.predict_proba(vect)[:, 1]

            vect = ide.transform(data)
            pred_ide = ide_model.predict_proba(vect)[:, 1]

            # Round the results
            out_tox = round(pred_tox[0], 2)
            out_sev = round(pred_sev[0], 2)
            out_obs = round(pred_obs[0], 2)
            out_ins = round(pred_ins[0], 2)
            out_thr = round(pred_thr[0], 2)
            out_ide = round(pred_ide[0], 2)

            # Prepare response
            response = {
                'original_text': text,
                'toxicity_results': {
                    "Prob (Toxic)": out_tox,
                    "Prob (Severe Toxic)": out_sev,
                    "Prob (Obscene)": out_obs,
                    "Prob (Insult)": out_ins,
                    "Prob (Threat)": out_thr,
                    "Prob (Identity Hate)": out_ide
                }
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": f"Error during prediction: {e}"}), 500

# Adding the resource to the API
api.add_resource(PredictToxicity, "/predict")

if __name__ == "__main__":
    app.run(debug=True)
