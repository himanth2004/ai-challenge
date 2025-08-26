from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
CORS(app)

# Lazy-load behavior model to avoid heavy imports at startup
behavior_model = None
_torch = None

def get_behavior_model():
    global behavior_model, _torch
    if behavior_model is not None:
        return behavior_model
    try:
        print("Initializing behavior model (lazy load)...")
        model_path = os.path.join(BASE_DIR, "behavior_model.pth")
        if not os.path.exists(model_path):
            print(f"⚠️  Model file not found: {model_path}")
            return None
        # Import torch and model definition only when needed
        import importlib
        _torch = importlib.import_module('torch')
        model_def = importlib.import_module('model_definition')
        MouseDynamicsClassifier = getattr(model_def, 'MouseDynamicsClassifier')
        mdl = MouseDynamicsClassifier()
        state = _torch.load(model_path, map_location=_torch.device("cpu"))
        mdl.load_state_dict(state)
        mdl.eval()
        behavior_model = mdl
        print("✅ Behavior model loaded successfully (lazy)")
        return behavior_model
    except Exception as e:
        print(f"❌ Error lazy-loading behavior model: {e}")
        behavior_model = None
        return None

# Load typing model and vectorizer
try:
    print("Loading typing model and vectorizer...")
    model_path = os.path.join(BASE_DIR, "typing_model.pkl")
    typing_model = joblib.load(model_path) if os.path.exists(model_path) else None
    if typing_model is None:
        print(f"⚠️  Typing model file not found: {model_path}")
    
    # Create TF-IDF vectorizer deterministically (no heavy fitting needed at startup)
    typing_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1,2)
    )
    # Fit vectorizer on a small fixed seed corpus to initialize vocabulary
    seed_corpus = [
        "this is a sample text for initializing vectorizer",
        "another example text containing different words",
        "human typing pattern example with natural language",
        "robot typing pattern example with predictable text"
    ]
    typing_vectorizer.fit(seed_corpus)
    print("✅ Typing vectorizer initialized.")
except Exception as e:
    print(f"❌ Error initializing typing pipeline: {e}")
    typing_model = None
    typing_vectorizer = None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    behavior_features = data.get("behavior")
    print(f"\n--- Prediction Request Received ---")                # Printed in terminal
    print(f"Input data: {behavior_features}")                     # Printed in terminal
    print("Processing with behavior model...")                    # Printed in terminal

    model = get_behavior_model()
    if not model:
        print("❌ No behavior model loaded.")
        return jsonify({"prediction": "No behavior model loaded."})

    if not behavior_features or not isinstance(behavior_features, list):
        print("❌ Invalid or missing behavior data.")
        return jsonify({"prediction": "Invalid or missing behavior data."})

    try:
        coords = behavior_features
        # Use only the latest mouse coordinate (single point)
        if len(coords) > 1:
            coords = [coords[-1]]
        behavior_tensor = _torch.tensor([coords], dtype=_torch.float32)
        with _torch.no_grad():
            output = model(behavior_tensor)
            pred_behavior = _torch.argmax(output, dim=1).item()
        print(f"✅ Behavior model output: {pred_behavior}")        # Printed in terminal
        return jsonify({"prediction": pred_behavior})
    except Exception as e:
        print(f"❌ Error during prediction: {e}")                  # Printed in terminal
        return jsonify({"prediction": "Error during prediction."})

@app.route("/predict_typing", methods=["POST"])
def predict_typing():
    data = request.json
    text_input = data.get("text")
    print(f"\n--- Typing Prediction Request Received ---")          # Printed in terminal
    print(f"Input text: {text_input}")                            # Printed in terminal
    print("Processing with typing model...")                      # Printed in terminal

    if not typing_model or not typing_vectorizer:
        print("❌ No typing model or vectorizer loaded.")
        return jsonify({"prediction": "No typing model loaded."})

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
        
        # Vectorize the text using TF-IDF (use transform, not fit_transform)
        text_vectorized = typing_vectorizer.transform([cleaned_text])
        print(f"Vectorized shape: {text_vectorized.shape}")        # Printed in terminal
        
        # Ensure we have exactly 5000 features by padding or truncating
        if text_vectorized.shape[1] < 5000:
            # Pad with zeros to reach 5000 features
            from scipy.sparse import hstack, csr_matrix
            padding = csr_matrix((1, 5000 - text_vectorized.shape[1]))
            text_vectorized = hstack([text_vectorized, padding])
            print(f"Padded to shape: {text_vectorized.shape}")
        elif text_vectorized.shape[1] > 5000:
            # Truncate to 5000 features
            text_vectorized = text_vectorized[:, :5000]
            print(f"Truncated to shape: {text_vectorized.shape}")
        
        # Make prediction
        prediction = typing_model.predict(text_vectorized)[0]
        print(f"✅ Typing model output: {prediction}")             # Printed in terminal
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        print(f"❌ Error during typing prediction: {e}")           # Printed in terminal
        import traceback
        traceback.print_exc()  # Print full error traceback
        return jsonify({"prediction": "Error during typing prediction."})

@app.route("/")
def index():
    print("✅ Index route accessed")
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)