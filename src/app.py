import os
import joblib
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import torch
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)
CORS(app)

# Resolve the model paths
behavior_model_path = os.path.join(BASE_DIR, "behavior_model.pkl")
typing_pipeline_path = os.path.join(BASE_DIR, "typing_pipeline.pkl")

# Load models safely
if os.path.exists(behavior_model_path):
    behavior_model = joblib.load(behavior_model_path)
    print(f"✅ Loaded behavior model from: {behavior_model_path}")
else:
    print(f"❌ behavior_model.pkl not found at: {behavior_model_path}")
    behavior_model = None

if os.path.exists(typing_pipeline_path):
    typing_pipeline = joblib.load(typing_pipeline_path)
    print(f"✅ Loaded typing pipeline from: {typing_pipeline_path}")
else:
    print(f"❌ typing_pipeline.pkl not found at: {typing_pipeline_path}")
    typing_pipeline = None

import numpy as np

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    behavior_features = data.get("behavior")
    print(f"\n--- Prediction Request Received ---")
    print(f"Input data: {behavior_features}")
    print("Processing with behavior model...")

    if behavior_model is None:
        print("❌ No behavior model loaded.")
        return jsonify({"prediction": "No behavior model loaded."})

    if not behavior_features or not isinstance(behavior_features, list):
        print("❌ Invalid or missing behavior data.")
        return jsonify({"prediction": "Invalid or missing behavior data."})

    try:
        coords = behavior_features

        # Extract simple features (similar to training)
        xs = np.array([p[0] for p in coords])
        ys = np.array([p[1] for p in coords])
        dx = np.diff(xs) if len(xs) > 1 else np.array([0])
        dy = np.diff(ys) if len(ys) > 1 else np.array([0])
        dist = np.sqrt(dx**2 + dy**2)
        features = [
            xs.mean(), xs.std(), xs.min(), xs.max(),
            ys.mean(), ys.std(), ys.min(), ys.max(),
            dist.mean(), dist.std(), dist.sum()
        ]

        # RandomForest expects 2D array
        features_array = np.array([features])
        pred_behavior = int(behavior_model.predict(features_array)[0])

        print(f"✅ Behavior model output: {pred_behavior}")
        return jsonify({"prediction": pred_behavior})

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"prediction": "Error during prediction."})

@app.route("/predict_typing", methods=["POST"])
def predict_typing():
    data = request.json
    text_input = data.get("text")
    print(f"\n--- Typing Prediction Request Received ---")          # Printed in terminal
    print(f"Input text: {text_input}")                            # Printed in terminal
    print("Processing with typing pipeline...")                      # Printed in terminal

    if typing_pipeline is None:
        print("❌ No typing pipeline loaded.")
        return jsonify({"prediction": "No typing pipeline loaded."})

    if not text_input or not isinstance(text_input, str):
        print("❌ Invalid or missing text data.")
        return jsonify({"prediction": "Invalid or missing text data."})

    try:
        # Text preprocessing (same as training)
        replacements = {
            "he's": "he is", "she's": "she is", "it's": "it is", "I'm": "I am",
            "you're": "you are", "we're": "we are", "they're": "they are",
            "he'll": "he will", "she'll": "she will", "it'll": "it will",
            "i'll": "i will", "you'll": "you will", "we'll": "we will",
            "they'll": "they will", "he'd": "he would", "she'd": "she would",
            "it'd": "it would", "i'd": "i would", "you'd": "you would",
            "we'd": "we would", "they'd": "they would", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "can't": "cannot",
            "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
            "mightn't": "might not", "mustn't": "must not", "aren't": "are not",
            "isn't": "is not", "wasn't": "was not", "weren't": "were not",
            "im": "i am", "u": "you"
        }
        
        def clean_text(text):
            text = text.lower()
            for k, v in replacements.items():
                text = text.replace(k, v)
            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text.strip()
        
        # Clean the input text
        cleaned_text = clean_text(text_input)
        print(f"Cleaned text: {cleaned_text}")                     # Printed in terminal
        
        # Use the saved sklearn pipeline directly
        prediction = typing_pipeline.predict([cleaned_text])[0]
        print(f"✅ Typing pipeline output: {prediction}")             # Printed in terminal
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        print(f"❌ Error during typing prediction: {e}")           # Printed in terminal
        import traceback
        traceback.print_exc()  # Print full error traceback
        return jsonify({"prediction": "Error during typing prediction."})



@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "behavior_model_loaded": behavior_model is not None,
        "typing_pipeline_loaded": typing_pipeline is not None,
        "base_dir": BASE_DIR,
    })

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/about.html', methods=["GET"])
def aboutpage():
    return render_template('about.html')

@app.route('/main.html', methods=["GET"])
def mainpage():
    return render_template('main.html')

@app.route('/index.html', methods=["GET"])
def homepage():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)