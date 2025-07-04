#!/usr/bin/env python3
"""
Simplified Plant Disease Recognition System
"""

import os
import sys
import time
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid

# Global variables to store loaded model and data
model = None
plant_disease = None

def load_model():
    """Load the trained model"""
    global model
    print("Loading model...")
    start_time = time.time()
    try:
        model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
        load_time = time.time() - start_time
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def load_data():
    """Load the plant disease data"""
    global plant_disease
    print("Loading plant disease data...")
    try:
        with open("plant_disease.json", 'r') as file:
            plant_disease = json.load(file)
        print(f"✓ Plant disease data loaded successfully ({len(plant_disease)} entries)")
        return True
    except Exception as e:
        print(f"✗ Error loading plant disease data: {e}")
        return False

def extract_features(image_path):
    """Extract features from image"""
    try:
        image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        feature = tf.keras.utils.img_to_array(image)
        feature = np.array([feature])
        return feature
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def model_predict(image_path):
    """Make prediction using the loaded model"""
    global model, plant_disease
    if model is None:
        return "Model not loaded"
    if plant_disease is None:
        return "Plant disease data not loaded"
    
    try:
        img = extract_features(image_path)
        if img is None:
            return "Error processing image"
        
        prediction = model.predict(img)
        prediction_label = plant_disease[prediction.argmax()]
        return prediction_label
    except Exception as e:
        print(f"Error in prediction: {e}")
        return f"Prediction error: {e}"

# Create Flask app
app = Flask(__name__)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        try:
            image = request.files['img']
            if image.filename == '':
                return render_template('home.html', error="No file selected")
            
            # Create uploadimages directory if it doesn't exist
            os.makedirs('uploadimages', exist_ok=True)
            
            # Save uploaded image
            temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
            image_path = f'{temp_name}_{image.filename}'
            image.save(image_path)
            print(f'Processing: {image_path}')
            
            # Make prediction
            prediction = model_predict(f'./{image_path}')
            
            return render_template('home.html', result=True, imagepath=f'/{image_path}', prediction=prediction)
        except Exception as e:
            print(f"Error in upload: {e}")
            return render_template('home.html', error=f"Upload error: {e}")
    else:
        return redirect('/')

def main():
    """Main startup function"""
    print("=" * 60)
    print("Plant Disease Recognition System - Simplified")
    print("=" * 60)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    # Load data
    if not load_data():
        print("❌ Failed to load plant disease data. Exiting.")
        sys.exit(1)
    
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