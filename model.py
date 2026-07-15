import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config

def preprocess_landmarks(landmarks_list):
    """
    Normalizes 21 3D hand landmarks to be translation-invariant (centered at wrist)
    and scale-invariant (scaled by maximum coordinate spread).
    Input: landmarks_list = [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20] (len=63)
    Output: normalized numpy array of shape (63,)
    """
    coords = np.array(landmarks_list).reshape(21, 3)
    
    # 1. Translate: Make wrist (index 0) the origin (0, 0, 0)
    wrist = coords[0]
    translated = coords - wrist
    
    # 2. Scale: Normalize by max absolute coordinate magnitude to make scale-invariant
    max_val = np.max(np.abs(translated))
    if max_val > 0:
        normalized = translated / max_val
    else:
        normalized = translated
        
    return normalized.flatten()

def save_model(model, label_map):
    """Saves the trained model and label mapping to disk."""
    data = {
        "model": model,
        "label_map": label_map
    }
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"Model saved successfully to {config.MODEL_PATH}")

def load_model():
    """Loads the model and label mapping from disk. Returns (model, label_map) or (None, None) if not found."""
    if not os.path.exists(config.MODEL_PATH):
        return None, None
    try:
        with open(config.MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["label_map"]
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def train_gesture_classifier():
    """
    Reads the CSV dataset, trains a Random Forest Classifier,
    saves the model, and returns the validation accuracy.
    """
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError("Dataset CSV not found. Please generate or collect data first.")
        
    # Load dataset
    data = []
    labels = []
    
    with open(config.DATASET_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            label = parts[0]
            features = [float(x) for x in parts[1:]]
            data.append(features)
            labels.append(label)
            
    if len(data) == 0:
        raise ValueError("Dataset is empty. Cannot train model.")
        
    X = np.array(data)
    y = np.array(labels)
    
    # Get unique labels and map them to their string representation
    unique_labels = sorted(list(set(y)))
    label_map = {i: label for i, label in enumerate(unique_labels)}
    
    # Map string labels to numeric for training
    label_to_id = {label: i for i, label in label_map.items()}
    y_numeric = np.array([label_to_id[label] for label in y])
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric)
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save model and mapping
    save_model(clf, label_map)
    
    return accuracy, label_map
