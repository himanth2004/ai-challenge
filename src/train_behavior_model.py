import os
import glob
import csv
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Feature extraction
# ----------------------------
def extract_features(window):
    xs = np.array([p[0] for p in window])
    ys = np.array([p[1] for p in window])
    dx = np.diff(xs)
    dy = np.diff(ys)
    dist = np.sqrt(dx**2 + dy**2)
    features = [
        xs.mean(), xs.std(), xs.min(), xs.max(),
        ys.mean(), ys.std(), ys.min(), ys.max(),
        dist.mean(), dist.std(), dist.sum()
    ]
    return features

def read_session_csv(path):
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            try:
                x = float(row[-2])
                y = float(row[-1])
                coords.append((x, y))
            except:
                continue
    return coords

def make_windows(coords, window_size=50, stride=25):
    windows = []
    for start in range(0, len(coords) - window_size + 1, stride):
        windows.append(coords[start:start + window_size])
    return windows

def prepare_features(human_dirs, robot_dirs):
    X, y = [], []
    for dirs, label in [(human_dirs, 0), (robot_dirs, 1)]:
        for folder in dirs:
            for file in glob.glob(os.path.join(folder, "*.csv")):
                coords = read_session_csv(file)
                for w in make_windows(coords):
                    if w:
                        X.append(extract_features(w))
                        y.append(label)
    return np.array(X), np.array(y)

# ----------------------------
# Train and save model
# ----------------------------
def train_and_save_model():
    human_dirs = [os.path.join(BASE_DIR, "User7")]
    robot_dirs = [os.path.join(BASE_DIR, "User9")]

    X, y = prepare_features(human_dirs, robot_dirs)
    if len(X) == 0:
        raise RuntimeError("No training samples found. Check CSV paths.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))

    # Save model
    out_path = os.path.join(BASE_DIR, "behavior_model.pkl")
    joblib.dump(clf, out_path)
    print(f"Saved trained behavior model to: {out_path}")

# ----------------------------
# Run training
# ----------------------------
if __name__ == "__main__":
    train_and_save_model()
