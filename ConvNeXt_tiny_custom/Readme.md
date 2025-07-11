


# 🍅 Tomato Disease Classification Model

This repository contains a TensorFlow-based deep learning model for classifying **11 types of tomato diseases** using leaf images. The model is built using the **ConvNeXt-Tiny** architecture with a custom dense layer and is saved in `.keras` format.

---

## 🧠 Model Overview

- **Model**: ConvNeXt-Tiny + Dense classifier
- **Framework**: TensorFlow / Keras
- **Input size**: 224x224 RGB image
- **Output classes**:
  - Bacterial_spot  
  - Early_blight  
  - Late_blight  
  - Leaf_Mold  
  - Septoria_leaf_spot  
  - Spider_mites Two-spotted_spider_mite  
  - Target_Spot  
  - Tomato_Yellow_Leaf_Curl_Virus  
  - Tomato_mosaic_virus  
  - healthy  
  - powdery_mildew  

---

## ⬇️ Download the Model

> 📂 **Model Drive Link**:  
> [Tomato Disease Model (Google Drive)](https://drive.google.com/drive/folders/1zGOrtabjTjOwtKsIX9GPbz7BnZjyTM8b?usp=sharing)

### 🔧 Instructions:
1. Open the link above.
2. Download the `.keras` file inside the folder.
3. Place it in your working directory or update the script with the correct path.

---

## 🚀 Usage Instructions

### 📦 Requirements

* Python 3.x
* TensorFlow ≥ 2.10
* NumPy
* Pillow

Install dependencies with:

```bash
pip install tensorflow pillow numpy
```

### ✅ Basic Prediction

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("path/to/model.keras")

# Preprocess image
img = Image.open("path/to/image.jpg").convert("RGB").resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Class labels
class_names = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
    'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus', 'healthy', 'powdery_mildew'
]

# Predict
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]
print("Predicted Disease:", predicted_class)
```

---

### 🔢 Top-N Predictions

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("path/to/model.keras")

# Preprocess image
img = Image.open("path/to/image.jpg").convert("RGB").resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Class labels
class_names = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
    'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus', 'healthy', 'powdery_mildew'
]
# Predict and get top-N
preds = model.predict(img_array)[0]
N = 3  # Number of top predictions to show

top_indices = preds.argsort()[-N:][::-1]  # Sort and reverse

print("Top-N Predictions:")
for idx in top_indices:
    print(f"{class_names[idx]}: {preds[idx]*100:.2f}%")
```

