1. System Overview
 The AI Sentinel project is designed to detect anomalies in user behavior patterns by
 monitoring mouse dynamics and typing rhythms. By leveraging lightweight machine
 learning models hosted in a Flask backend, the system performs real-time classification
 while prioritizing low latency and privacy. The solution captures raw user interactions in the
 client browser, processes them into structured features, and evaluates them against
 trained models to detect unusual or fraudulent behavior.
2. High-Level Architecture
 The system is divided into four primary layers: 1. Client Layer: Captures raw interactions.
 2. Flask Application: Handles routing, endpoints, and serves templates. 3. Models Layer:
 Stores serialized machine learning artifacts. 4. Feature Extraction: Processes raw inputs
 into features.
 3. Core Components:
Client (Browser UI): Static pages under src/templates/ rendered by Flask. Provides interaction surfaces for mouse and typing tests. - Flask Application (src/app.py): - Serves HTML pages via Jinja2 templates. - Exposes REST endpoints: /health, /predict, /predict_typing. - Loads serialized models at startup using joblib. - Models (Serialized Artifacts): - behavior_model.pkl: RandomForest (or similar sklearn estimator) trained on mouse trajectory features. - typing_pipeline.pkl: Sklearn Pipeline (e.g., text vectorizer + classifier) for typing analysis. - Feature Extraction (Runtime): - Mouse trajectories: compute summary statistics over coordinate deltas and distances. - Text: lowercase, expand common contractions, remove non-letters before pipeline inference.
Component	Description
Client (Browser UI)
	Static HTML pages that allow users to perform typing/mouse tests. Implemented with JavaScript listeners.
Flask Application
	Backend service serving templates and exposing endpoints for predictions.

Models
	Pre-trained models serialized via joblib: RandomForest for mouse dynamics, sklearn Pipeline for typing.

Feature Extraction
	Runtime scripts for deriving trajectory statistics and text cleaning.


4. Data Flow Architecture
 The data flow begins at the client when the user interacts with the mouse or keyboard.
 JavaScript listeners capture these events and package them as JSON payloads to be sent
 to the Flask backend. At the backend, the data is validated, preprocessed, and
 transformed into features suitable for the ML model. Predictions are generated and
 returned to the client as JSON responses. Historical logs may optionally be stored in a
 database for further analysis.
 

5. Request Lifecycle
 1. Client sends request with captured data. 2. Flask validates input format. 3. Features are
 extracted using appropriate preprocessing modules. 4. Model inference is performed. 5.
 Flask returns JSON response with prediction.
 6. Endpoints & Data Contracts- `GET /` → Returns index.html - `GET /main.html` → Returns interaction page - `POST
 /predict` → Accepts JSON with mouse behavior - `POST /predict_typing` → Accepts
 JSON with text
