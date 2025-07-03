#!/usr/bin/env python3
"""
Plant Disease Recognition System - Startup Script
"""

import sys
import time
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import os

def load_model():
    """Load the trained model"""
    print("Loading model...")
    start_time = time.time()
    try:
        model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
        load_time = time.time() - start_time
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def load_data():
    """Load the plant disease data"""
    print("Loading plant disease data...")
    try:
        with open("plant_disease.json", 'r') as file:
            plant_disease = json.load(file)
        print(f"✓ Plant disease data loaded successfully ({len(plant_disease)} entries)")
        return plant_disease
    except Exception as e:
        print(f"✗ Error loading plant disease data: {e}")
        return {}

def create_app(model, plant_disease):
    """Create and configure the Flask app"""
    print("Creating Flask application...")
    
    app = Flask(__name__)
    
    # Label mapping
    label = ['Apple___Apple_scab',
     'Apple___Black_rot',
     'Apple___Cedar_apple_rust',
     'Apple___healthy',
     'Background_without_leaves',
     'Blueberry___healthy',
     'Cherry___Powdery_mildew',
     'Cherry___healthy',
     'Corn___Cercospora_leaf_spot Gray_leaf_spot',
     'Corn___Common_rust',
     'Corn___Northern_Leaf_Blight',
     'Corn___healthy',
     'Grape___Black_rot',
     'Grape___Esca_(Black_Measles)',
     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
     'Grape___healthy',
     'Orange___Haunglongbing_(Citrus_greening)',
     'Peach___Bacterial_spot',
     'Peach___healthy',
     'Pepper,_bell___Bacterial_spot',
     'Pepper,_bell___healthy',
     'Potato___Early_blight',
     'Potato___Late_blight',
     'Potato___healthy',
     'Raspberry___healthy',
     'Soybean___healthy',
     'Squash___Powdery_mildew',
     'Strawberry___Leaf_scorch',
     'Strawberry___healthy',
     'Tomato___Bacterial_spot',
     'Tomato___Early_blight',
     'Tomato___Late_blight',
     'Tomato___Leaf_Mold',
     'Tomato___Septoria_leaf_spot',
     'Tomato___Spider_mites Two-spotted_spider_mite',
     'Tomato___Target_Spot',
     'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
     'Tomato___Tomato_mosaic_virus',
     'Tomato___healthy']

    @app.route('/uploadimages/<path:filename>')
    def uploaded_images(filename):
        return send_from_directory('./uploadimages', filename)

    @app.route('/', methods=['GET'])
    def home():
        return render_template('home.html')

    def extract_features(image):
        image = tf.keras.utils.load_img(image, target_size=(160, 160))
        feature = tf.keras.utils.img_to_array(image)
        feature = np.array([feature])
        return feature

    def model_predict(image):
        if model is None:
            return "Model not loaded"
        img = extract_features(image)
        prediction = model.predict(img)
        prediction_label = plant_disease[prediction.argmax()]
        return prediction_label

    @app.route('/upload/', methods=['POST', 'GET'])
    def uploadimage():
        if request.method == "POST":
            image = request.files['img']
            temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
            image.save(f'{temp_name}_{image.filename}')
            print(f'Processing: {temp_name}_{image.filename}')
            prediction = model_predict(f'./{temp_name}_{image.filename}')
            return render_template('home.html', result=True, imagepath=f'/{temp_name}_{image.filename}', prediction=prediction)
        else:
            return redirect('/')

    print("✓ Flask application created successfully")
    return app

def main():
    """Main startup function"""
    print("=" * 60)
    print("Plant Disease Recognition System - Startup")
    print("=" * 60)
    
    # Load model
    model = load_model()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    # Load data
    plant_disease = load_data()
    if not plant_disease:
        print("❌ Failed to load plant disease data. Exiting.")
        sys.exit(1)
    
    # Create app
    app = create_app(model, plant_disease)
    
    print("\n" + "=" * 60)
    print("🚀 Starting Flask server...")
    print("📱 Access the application at: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 