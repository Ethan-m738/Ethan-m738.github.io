from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import io
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/')
def hello_world():
    return render_template("index.html")

# Load TensorFlow model and labels
model_path = '/home/winteriscoming/mysite/dataset'
model = tf.saved_model.load(model_path)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract fields from the JSON data
        age = data.get('Age')
        course = data.get('Course')
        gender = data.get('Gender')
        cgpa = data.get('CGPA')
        stress_level = data.get('Stress_Level')
        anxiety_score = data.get('Anxiety_Score')
        sleep_quality = data.get('Sleep_Quality')
        physical_activity = data.get('Physical_Activity')
        diet_quality = data.get('Diet_Quality')
        social_support = data.get('Social_Support')
        relationship_status = data.get('Relationship_Status')
        substance_use = data.get('Substance_Use')
        counseling_service_use = data.get('Counseling_Service_Use')
        family_history = data.get('Family_History')
        chronic_illness = data.get('Chronic_Illness')
        financial_stress = data.get('Financial_Stress')
        extracurricular_involvement = data.get('Extracurricular_Involvement')
        semester_credit_load = data.get('Semester_Credit_Load')
        residence_type = data.get('Residence_Type')

        # Validate required fields
        required_fields = [age, course, gender, cgpa, stress_level, anxiety_score, sleep_quality, physical_activity,
                           diet_quality, social_support, relationship_status, substance_use, counseling_service_use,
                           family_history, chronic_illness, financial_stress, extracurricular_involvement,
                           semester_credit_load, residence_type]
        if None in required_fields:
            return jsonify({'error': 'All fields must be provided'}), 400

        # Convert categorical variables to numerical format (if necessary)
        # For simplicity, assume these are one-hot encoded or already numerical.
        # You may need to apply more sophisticated preprocessing based on your model requirements.

        inputs = np.array([[age, course, gender, cgpa, stress_level, anxiety_score, sleep_quality, physical_activity,
                            diet_quality, social_support, relationship_status, substance_use, counseling_service_use,
                            family_history, chronic_illness, financial_stress, extracurricular_involvement,
                            semester_credit_load, residence_type]], dtype=np.float32)

        tensor = tf.convert_to_tensor(inputs)

        # Perform inference directly using model
        # Ensure your model is compatible with this approach
        predictions = model(tensor)

        # Adjust the key to match the model's output
        # If your model outputs a dictionary, replace 'output_0' with the appropriate key
        prediction_result = predictions['output_0'].numpy() if isinstance(predictions, dict) else predictions.numpy()

        return jsonify({'prediction': float(prediction_result[0][0])})


    except Exception as e:
        # Handle exceptions and return an error message
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
