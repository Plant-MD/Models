import tensorflow as tf
import json

print("Loading model...")
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
print("Model loaded successfully!")

print("Loading JSON data...")
with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)
print("JSON data loaded successfully!")

print("Model summary:")
model.summary() 