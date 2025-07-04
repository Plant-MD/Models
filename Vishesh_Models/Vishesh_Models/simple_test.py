import tensorflow as tf
import json
import time
import os

print("=== Plant Disease Recognition System Test ===")
print(f"TensorFlow version: {tf.__version__}")

# Test 1: Check if model file exists
print("\n1. Checking model file...")
model_path = "models/plant_disease_recog_model_pwp.keras"
if os.path.exists(model_path):
    print(f"✓ Model file exists: {model_path}")
    print(f"  File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
else:
    print(f"✗ Model file not found: {model_path}")

# Test 2: Load model
print("\n2. Loading model...")
start_time = time.time()
try:
    model = tf.keras.models.load_model(model_path)
    load_time = time.time() - start_time
    print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
    print(f"  Model type: {type(model)}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ Error loading model: {e}")

# Test 3: Load JSON data
print("\n3. Loading JSON data...")
try:
    with open("plant_disease.json", 'r') as file:
        plant_disease = json.load(file)
    print(f"✓ JSON data loaded successfully")
    print(f"  Number of entries: {len(plant_disease)}")
except Exception as e:
    print(f"✗ Error loading JSON: {e}")

# Test 4: Test Flask import
print("\n4. Testing Flask import...")
try:
    from flask import Flask
    app = Flask(__name__)
    print("✓ Flask imported and app created successfully")
except Exception as e:
    print(f"✗ Error with Flask: {e}")

print("\n=== Test Complete ===") 