from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app) # Allows the frontend to communicate with backend

# Load your specific model
model = tf.keras.models.load_model("plant_model.keras", compile=False)

class_names = [
    "Pepper__Bacterial_spot", "Pepper__healthy", "Potato__Early_blight",
    "Potato__healthy", "Potato__Late_blight", "Tomato__Bacterial_spot",
    "Tomato__Early_blight", "Tomato__healthy", "Tomato__Late_blight",
    "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot", "Tomato__Spider_mites",
    "Tomato__Target_Spot", "Tomato__Yellow_Leaf_Curl_Virus",
    "Tomato__mosaic_virus", "Other"
]

descriptions = {
    "Potato__Early_blight": "Fungal disease. Use fungicide and remove infected leaves.",
    "Potato__Late_blight": "Serious disease. Remove plant and apply copper fungicide.",
    "Tomato__Early_blight": "Brown spots on leaves. Use fungicide spray.",
    "Tomato__healthy": "Plant is healthy.",
    "Pepper__healthy": "Plant is healthy."
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        # img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (224, 224 ))
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image file."}), 400
        img = cv2.resize(img, (224, 224))
        
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]
        index = np.argmax(pred)
        disease = class_names[index]

        return jsonify({
            "disease": disease,
            "confidence": float(pred[index]),
            "description": descriptions.get(disease, "Consult a specialist for treatment steps.")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
