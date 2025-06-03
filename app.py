import os
os.environ['MPLBACKEND'] = 'agg'

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# Load the best model
model = tf.keras.models.load_model('/Users/vigneshs/Desktop/Neem_leaf_project/model_final/best_model.keras')

# Define class labels
class_labels = sorted(['Alternaria', 'Dieback', 'Healthy', 'Leaf_Blight', 'Leaf_Miners', 'Leaf_Rust', 'Powdery_Mildew'])

# Prediction function
def predict_neem_disease(image):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx] * 100)
    
    return predicted_class, confidence

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Load image from request
    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400
    
    # Get prediction
    predicted_class, confidence = predict_neem_disease(img)
    
    # Return result
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)