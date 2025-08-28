

# 1. Model Architecture and Training

## Mouse Dynamics

The mouse dynamics model is trained on CSV files containing user
interaction data (record timestamp, client timestamp, button, state, x,
y). Sliding windows of 50 points with a stride of 25 are used to extract
meaningful patterns.\
\
Extracted features include:\
- Mean, standard deviation, minimum, and maximum of x and y positions.\
- Distance-based features such as mean, standard deviation, and sum of
consecutive point distances.\
\
A Random Forest classifier with 100 estimators is trained with a 90/10
train-validation split. The final model outputs binary predictions (0 =
human, 1 = robot) and is saved as \'behavior_model.pkl\'.

## Typing Patterns

The typing model is trained on separate human (4,727 lines) and robot
(2,364 lines) conversation datasets. Text is preprocessed by converting
to lowercase, expanding contractions (e.g., don\'t → do not), and
removing non-alphabetic characters.\
\
TF-IDF vectorization is used with the following configurations:\
- English stop words removed.\
- Maximum 5000 features.\
- Unigram and bigram range.\
\
A Logistic Regression model with a maximum of 2000 iterations is trained
using an 80/20 split. The pipeline (TF-IDF → Logistic Regression) is
saved as \'typing_pipeline.pkl\'. The output is binary classification (0
= human, 1 = robot).

# 2. Feature Engineering Specifics

Mouse Features:\
- Capture human variability in movements compared to robotic precision.\
- Humans typically exhibit curved, inconsistent movements.\
- Bots tend to produce straight, uniform patterns.\
\
Text Features:\
- TF-IDF emphasizes unique vocabulary and stylistic patterns.\
- Cleaning normalizes input but retains writing style signals.\
- Bigrams capture contextual and phrase-level differences.

# 3. API Implementation

Mouse Prediction (/predict):\
- Validates input coordinate data.\
- Extracts features consistent with training.\
- Performs inference using the Random Forest model.\
- Returns JSON with prediction (0 = human, 1 = robot).\
- Includes error handling for invalid inputs or missing models.\
\
Typing Prediction (/predict_typing):\
- Validates text string input.\
- Applies preprocessing and TF-IDF transformation.\
- Logistic Regression inference generates predictions.\
- Returns JSON with prediction (0 = human, 1 = robot).\
- Includes error handling for invalid inputs or missing pipeline.

# 4. Frontend Implementation

Mouse Monitoring:\
- Mousemove events tracked in real-time.\
- Data buffered with 1-second inactivity timeout.\
- Provides live visual feedback of coordinates.\
- Start/Stop controls available with status indicators.\
\
Typing Analysis:\
- Text input supports live character count and WPM estimation.\
- If WPM \> 250, flagged as robot immediately.\
- Otherwise, backend model validates typing behavior.\
\
Results Display:\
- Separate cards for mouse and typing analysis.\
- Processing time shown for transparency.\
- Demo confidence proxy (80--100%).\
- System log records activities with timestamps.

# 5. Model Loading and Management

Models are safely loaded at startup with existence checks. A health
endpoint ensures models are available and running. Graceful error
handling ensures the system degrades smoothly if files are missing.

# 6. Data Flow Implementation

1\. Frontend captures user input in real-time.\
2. Data is packaged into JSON and sent to backend.\
3. Backend extracts features and runs inference.\
4. Response is returned immediately to the client.\
5. Results and logs are displayed in the frontend.

📌 Diagram Placeholder: Data Flow Diagram

# 7. Performance Characteristics

\- Typical requests processed in sub-second times.\
- Simple statistical feature extraction ensures efficiency.\
- Models are compact and memory-friendly.\
- Stateless API design supports horizontal scalability.

# 8. Error Handling and Robustness

\- Input validation enforced at every endpoint.\
- Graceful fallback in case of missing models.\
- Comprehensive logging for transparency.\
- UI displays friendly error messages.

# 9. Deployment Considerations

\- Models stored as lightweight .pkl files.\
- Dependencies limited to Flask, scikit-learn, NumPy.\
- Containerization-ready stateless API.\
- Operational monitoring supported through health checks.\
- CORS enabled for secure web access.\
\
This project has been deployed successfully on Render. The live
deployment can be accessed here:\
🔗<https://samsung-ai-challenge.onrender.com/>
