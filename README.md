# 🤟 Modern Sign Language Translator & Gesture Studio

A premium, modular Python desktop application for real-time sign language translation. The application uses **OpenCV** to stream camera video, **MediaPipe** to extract 21 3D hand joint landmarks, and a **Random Forest Classifier** (`scikit-learn`) to translate hand positions in real-time.

It features a custom glassmorphic Dark interface built using `customtkinter`, allowing users to:
1. **Translate gestures in real-time** with a visual 3D skeleton overlays on screen.
2. **Collect custom gesture data** directly through the camera.
3. **Train the translation engine** interactively in the GUI.
4. **Speak translated text** out loud using automated Text-to-Speech (TTS).

---

## 🚀 Quick Start

Ensure you have [uv](https://github.com/astral-sh/uv) installed. Then run the commands below.

### 1. Initialize and Install Dependencies
Install all package dependencies in a virtual environment:
```bash
uv pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data (First Run Only)
Before launching the app, run the generator script. This creates a synthetic dataset modeling 5 standard hand shapes (Fist, Open Hand, Victory, Thumbs Up, I Love You) and trains the initial model (`gesture_model.pkl`), ensuring the app works out-of-the-box:
```bash
uv run generator.py
```

### 3. Run the Application
Launch the graphical interface:
```bash
uv run app.py
```

---

## 🎨 Studio Walkthrough

### 1. Live Translation
- Position your hand in front of the camera. The app automatically detects 21 hand landmarks and outlines the skeleton.
- If a gesture matches a trained pattern, it will display the label and confidence score.
- Stable predictions are added to the **Translation Stream**.
- Click **Speak Output** to read the translated sentence aloud.

### 2. Custom Gesture Creator
- Enter a name for your custom gesture (e.g., "Peace", "Ok").
- Click **🔴 Start Gesture Recording**.
- The progress bar will fill up as the app records 80 frames of your hand configuration. Move your hand slightly during recording to capture variations.
- Once done, click **⚙️ Train Translation Model** to rebuild the classifier. It will take only a second and output the new model accuracy!
