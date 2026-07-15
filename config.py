import os

# Base Directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# File Paths
MODEL_PATH = os.path.join(DATA_DIR, "gesture_model.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "landmarks_dataset.csv")

# Camera & Recognition Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RECORDING_FRAMES = 80  # Number of frames to record for a single gesture
CONFIDENCE_THRESHOLD = 0.75  # Prediction probability threshold

# UI Colors & Theme ( Sleek Modern Dark Theme )
THEME_MODE = "Dark"
COLOR_THEME = "blue"

COLORS = {
    "bg_dark": "#0B0F19",       # Deep dark space blue
    "card_dark": "#161D30",     # Card background
    "accent": "#3B82F6",        # Neon blue accent
    "accent_hover": "#2563EB",  # Darker accent for hover
    "success": "#10B981",       # Emerald green
    "warning": "#F59E0B",       # Amber warning
    "danger": "#EF4444",        # Rose red
    "text_primary": "#F3F4F6",  # Cool gray/white
    "text_secondary": "#9CA3AF" # Muted gray
}

# Default programmatically generated gestures
DEFAULT_GESTURES = {
    0: "Fist",
    1: "Open Hand",
    2: "Victory",
    3: "Thumbs Up",
    4: "I Love You"
}
