from flask import Flask, request, jsonify
from flask_cors import CORS
from ML_Predictor import Predictor, PredictorError


# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize predictor object
predictor = Predictor(folderPath='C:/Users/hawia/MayoTest', createModel=False, createCleaner=False)

# Define route for prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from request
        data = request.get_json()

        # Check data format and Encounter ID presence
        if not data or 'id' not in data:
            return jsonify({'error': 'Invalid data format'}), 400

        # Extract Encounter ID from data
        encounter_id = data['id']

        # Check if Encounter ID is valid
        if not predictor.SHOWCASEONLY_findPatient(encounter_id):
            return jsonify({'error': 'Invalid EncounterID'}), 400
        
        # Predict transitional care probability
        probability = round(100 * predictor.predictPatient(), 2)
        print(probability)  # Print probability for debugging
        return jsonify({'probability': probability}), 200

    except PredictorError as e:
        # Handle predictor-specific errors
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        # Handle general server errors
        return jsonify({'error': 'Internal server error'}), 500

# Run the Flask application if this script is executed directly
if __name__ == "__main__":
    app.run()
