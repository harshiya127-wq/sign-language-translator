import os
import numpy as np
import config
import model

def get_base_hand(finger_states):
    """
    Constructs a base 3D hand structure with 21 landmarks based on finger states.
    finger_states: list of booleans [thumb, index, middle, ring, pinky] (True = extended, False = folded)
    """
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_states
    
    # 21 landmarks, initialized to zeros
    coords = np.zeros((21, 3))
    
    # Landmark 0: Wrist
    coords[0] = [0.0, 0.0, 0.0]
    
    # Landmark 1-4: Thumb
    coords[1] = [0.15, 0.1, -0.05]
    if thumb_ext:
        coords[2] = [0.28, 0.18, -0.08]
        coords[3] = [0.36, 0.24, -0.1]
        coords[4] = [0.42, 0.28, -0.11]
    else:
        coords[2] = [0.18, 0.15, 0.05]
        coords[3] = [0.15, 0.18, 0.08]
        coords[4] = [0.1, 0.18, 0.05]
        
    # Landmark 5-8: Index finger
    coords[5] = [0.1, 0.4, 0.0]
    if index_ext:
        coords[6] = [0.1, 0.58, 0.0]
        coords[7] = [0.1, 0.70, 0.0]
        coords[8] = [0.1, 0.78, 0.0]
    else:
        coords[6] = [0.1, 0.35, 0.08]
        coords[7] = [0.08, 0.28, 0.12]
        coords[8] = [0.07, 0.26, 0.08]
        
    # Landmark 9-12: Middle finger
    coords[9] = [0.02, 0.42, 0.0]
    if middle_ext:
        coords[10] = [0.02, 0.60, 0.0]
        coords[11] = [0.02, 0.72, 0.0]
        coords[12] = [0.02, 0.80, 0.0]
    else:
        coords[10] = [0.02, 0.37, 0.08]
        coords[11] = [0.02, 0.30, 0.12]
        coords[12] = [0.02, 0.28, 0.08]
        
    # Landmark 13-16: Ring finger
    coords[13] = [-0.05, 0.4, 0.0]
    if ring_ext:
        coords[14] = [-0.05, 0.58, 0.0]
        coords[15] = [-0.05, 0.70, 0.0]
        coords[16] = [-0.05, 0.78, 0.0]
    else:
        coords[14] = [-0.05, 0.35, 0.08]
        coords[15] = [-0.05, 0.28, 0.12]
        coords[16] = [-0.05, 0.26, 0.08]
        
    # Landmark 17-20: Pinky finger
    coords[17] = [-0.12, 0.35, 0.0]
    if pinky_ext:
        coords[18] = [-0.12, 0.53, 0.0]
        coords[19] = [-0.12, 0.65, 0.0]
        coords[20] = [-0.12, 0.73, 0.0]
    else:
        coords[18] = [-0.12, 0.30, 0.08]
        coords[19] = [-0.12, 0.24, 0.12]
        coords[20] = [-0.12, 0.22, 0.08]
        
    return coords

def rotate_hand(coords, roll_deg, pitch_deg, yaw_deg):
    """Applies roll, pitch, and yaw rotations to 3D hand coordinates."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    
    # Rotation matrix around Z (roll)
    R_z = np.array([
        [np.cos(r), -np.sin(r), 0],
        [np.sin(r), np.cos(r), 0],
        [0, 0, 1]
    ])
    
    # Rotation matrix around X (pitch)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p), np.cos(p)]
    ])
    
    # Rotation matrix around Y (yaw)
    R_y = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    # Combined rotation matrix
    R = R_z @ R_x @ R_y
    return coords @ R

def generate_synthetic_data(samples_per_gesture=200):
    """
    Generates synthetic landmark configurations for standard gestures,
    normalizes them, and saves them to the CSV dataset file.
    """
    print("Generating synthetic hand gestures data...")
    
    # Define finger states for our default gestures: [thumb, index, middle, ring, pinky]
    gesture_states = {
        "Fist": [False, False, False, False, False],
        "Open Hand": [True, True, True, True, True],
        "Victory": [False, True, True, False, False],
        "Thumbs Up": [True, False, False, False, False],
        "I Love You": [True, True, False, False, True]
    }
    
    dataset_records = []
    
    for label, states in gesture_states.items():
        base_hand = get_base_hand(states)
        
        for _ in range(samples_per_gesture):
            # 1. Randomly rotate hand within reasonable ranges
            roll = np.random.uniform(-40, 40)
            pitch = np.random.uniform(-30, 30)
            yaw = np.random.uniform(-35, 35)
            rotated = rotate_hand(base_hand, roll, pitch, yaw)
            
            # 2. Randomly scale the hand coordinates
            scale = np.random.uniform(0.7, 1.3)
            scaled = rotated * scale
            
            # 3. Add random gaussian noise to simulate measurement errors
            noise = np.random.normal(0, 0.015, scaled.shape)
            noisy_hand = scaled + noise
            
            # 4. Normalize the hand to make it translation & scale invariant
            flat_hand = noisy_hand.flatten().tolist()
            normalized_hand = model.preprocess_landmarks(flat_hand)
            
            # 5. Format row: label, x0, y0, z0, ..., x20, y20, z20
            record = [label] + normalized_hand.tolist()
            dataset_records.append(record)
            
    # Write to CSV file
    os.makedirs(os.path.dirname(config.DATASET_PATH), exist_ok=True)
    with open(config.DATASET_PATH, "w") as f:
        for record in dataset_records:
            line = ",".join([str(record[0])] + [f"{val:.6f}" for val in record[1:]])
            f.write(line + "\n")
            
    print(f"Dataset generated with {len(dataset_records)} samples at {config.DATASET_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()
    print("Training initial classifier model...")
    acc, labels = model.train_gesture_classifier()
    print(f"Initial model trained with validation accuracy: {acc:.2%}")
    print(f"Label Map: {labels}")
