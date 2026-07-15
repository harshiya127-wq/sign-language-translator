import os
import time
import threading
import queue
import cv2
import numpy as np
import customtkinter as ctk
import pyttsx3
from PIL import Image, ImageTk
import mediapipe as mp

import config
import model

# Set modern look
ctk.set_appearance_mode(config.THEME_MODE)
ctk.set_default_color_theme(config.COLOR_THEME)

class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure Window
        self.title("Sign Language Translator & Gesture Studio")
        self.geometry("1100x700")
        self.configure(fg_color=config.COLORS["bg_dark"])
        
        # Load Model
        self.clf, self.label_map = model.load_model()
        
        # App State
        self.camera_active = False
        self.recording_mode = False
        self.recorded_frames_count = 0
        self.recording_label_name = ""
        self.recording_buffer = []
        
        # Translation Smoothing State
        self.prediction_history = []
        self.last_translated_word = ""
        self.stable_frames_required = 10
        self.last_speech_time = 0
        
        # Communication Queues
        self.frame_queue = queue.Queue(maxsize=1)
        self.recording_status_queue = queue.Queue()
        
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Design Layout
        self.setup_ui()
        
        # Start Video Capture Thread
        self.start_camera()
        
        # Start GUI polling for frame updates
        self.update_gui_loop()
        
    def setup_ui(self):
        # Grid Configuration (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=6) # Camera Column
        self.grid_columnconfigure(1, weight=4) # Sidebar/Controls Column
        
        # --- LEFT PANEL: CAMERA VIEW ---
        self.camera_panel = ctk.CTkFrame(self, fg_color=config.COLORS["card_dark"], corner_radius=15)
        self.camera_panel.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.camera_panel.grid_rowconfigure(0, weight=1)
        self.camera_panel.grid_rowconfigure(1, weight=0)
        self.camera_panel.grid_columnconfigure(0, weight=1)
        
        # Webcam Feed Label (Fallback to dark placeholder)
        self.video_display = ctk.CTkLabel(
            self.camera_panel, 
            text="Initializing Camera Feed...", 
            font=("Outfit", 18, "bold"),
            text_color=config.COLORS["text_secondary"]
        )
        self.video_display.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Webcam Bottom Info Bar
        self.info_bar = ctk.CTkFrame(self.camera_panel, fg_color="transparent")
        self.info_bar.grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        self.info_bar.grid_columnconfigure(0, weight=1)
        self.info_bar.grid_columnconfigure(1, weight=1)
        
        self.camera_status_lbl = ctk.CTkLabel(
            self.info_bar,
            text="Status: Connecting...",
            font=("Outfit", 13, "bold"),
            text_color=config.COLORS["warning"]
        )
        self.camera_status_lbl.grid(row=0, column=0, sticky="w")
        
        self.current_gesture_lbl = ctk.CTkLabel(
            self.info_bar,
            text="Detected: --",
            font=("Outfit", 16, "bold"),
            text_color=config.COLORS["accent"]
        )
        self.current_gesture_lbl.grid(row=0, column=1, sticky="e")
        
        # --- RIGHT PANEL: TRANSLATION & CONTROLS ---
        self.sidebar = ctk.CTkFrame(self, fg_color="transparent")
        self.sidebar.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.sidebar.grid_rowconfigure(0, weight=5) # History Card
        self.sidebar.grid_rowconfigure(1, weight=3) # Recording Card
        self.sidebar.grid_rowconfigure(2, weight=2) # Training Card
        self.sidebar.grid_columnconfigure(0, weight=1)
        
        # CARD 1: TRANSLATION HISTORY
        self.history_card = ctk.CTkFrame(self.sidebar, fg_color=config.COLORS["card_dark"], corner_radius=15)
        self.history_card.grid(row=0, column=0, pady=(0, 10), sticky="nsew")
        self.history_card.grid_rowconfigure(1, weight=1)
        self.history_card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            self.history_card,
            text="TRANSLATION STREAM",
            font=("Outfit", 14, "bold"),
            text_color=config.COLORS["accent"]
        ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")
        
        self.history_text = ctk.CTkTextbox(
            self.history_card,
            font=("Consolas", 16),
            fg_color=config.COLORS["bg_dark"],
            text_color=config.COLORS["text_primary"],
            corner_radius=8
        )
        self.history_text.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")
        
        self.history_actions = ctk.CTkFrame(self.history_card, fg_color="transparent")
        self.history_actions.grid(row=2, column=0, padx=15, pady=(5, 15), sticky="ew")
        self.history_actions.grid_columnconfigure(0, weight=1)
        self.history_actions.grid_columnconfigure(1, weight=1)
        
        self.speak_btn = ctk.CTkButton(
            self.history_actions,
            text="🔊 Speak Output",
            font=("Outfit", 13, "bold"),
            fg_color=config.COLORS["accent"],
            hover_color=config.COLORS["accent_hover"],
            command=self.speak_history
        )
        self.speak_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        
        self.clear_btn = ctk.CTkButton(
            self.history_actions,
            text="🗑️ Clear",
            font=("Outfit", 13, "bold"),
            fg_color="transparent",
            border_color=config.COLORS["text_secondary"],
            border_width=1,
            hover_color="#1E293B",
            command=self.clear_history
        )
        self.clear_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        # CARD 2: GESTURE CREATOR STUDIO
        self.record_card = ctk.CTkFrame(self.sidebar, fg_color=config.COLORS["card_dark"], corner_radius=15)
        self.record_card.grid(row=1, column=0, pady=10, sticky="nsew")
        
        ctk.CTkLabel(
            self.record_card,
            text="GESTURE RECORDING STUDIO",
            font=("Outfit", 14, "bold"),
            text_color=config.COLORS["accent"]
        ).pack(anchor="w", padx=15, pady=(15, 5))
        
        self.gesture_entry = ctk.CTkEntry(
            self.record_card,
            placeholder_text="Enter Gesture Name (e.g. 'Hello')",
            font=("Outfit", 13),
            fg_color=config.COLORS["bg_dark"],
            text_color=config.COLORS["text_primary"],
            border_color=config.COLORS["text_secondary"],
            height=35
        )
        self.gesture_entry.pack(fill="x", padx=15, pady=8)
        
        self.record_btn = ctk.CTkButton(
            self.record_card,
            text="🔴 Start Gesture Recording",
            font=("Outfit", 13, "bold"),
            fg_color=config.COLORS["danger"],
            hover_color="#DC2626",
            height=35,
            command=self.toggle_recording
        )
        self.record_btn.pack(fill="x", padx=15, pady=5)
        
        self.record_progress = ctk.CTkProgressBar(
            self.record_card,
            progress_color=config.COLORS["success"],
            fg_color=config.COLORS["bg_dark"],
            height=8
        )
        self.record_progress.pack(fill="x", padx=15, pady=8)
        self.record_progress.set(0)
        
        self.record_status_lbl = ctk.CTkLabel(
            self.record_card,
            text="Enter a name and hit Record to build custom gestures.",
            font=("Outfit", 11),
            text_color=config.COLORS["text_secondary"]
        )
        self.record_status_lbl.pack(anchor="w", padx=15, pady=(0, 10))
        
        # CARD 3: MODEL TRAINER
        self.train_card = ctk.CTkFrame(self.sidebar, fg_color=config.COLORS["card_dark"], corner_radius=15)
        self.train_card.grid(row=2, column=0, pady=(10, 0), sticky="nsew")
        
        ctk.CTkLabel(
            self.train_card,
            text="MODEL ENGINE MANAGEMENT",
            font=("Outfit", 14, "bold"),
            text_color=config.COLORS["accent"]
        ).pack(anchor="w", padx=15, pady=(15, 5))
        
        self.train_btn = ctk.CTkButton(
            self.train_card,
            text="⚙️ Train Translation Model",
            font=("Outfit", 13, "bold"),
            fg_color=config.COLORS["success"],
            hover_color="#059669",
            height=38,
            command=self.train_model_async
        )
        self.train_btn.pack(fill="x", padx=15, pady=5)
        
        self.model_status_lbl = ctk.CTkLabel(
            self.train_card,
            text="Model Ready." if self.clf else "No Model Found. Click train or record custom gestures.",
            font=("Outfit", 12),
            text_color=config.COLORS["success"] if self.clf else config.COLORS["warning"]
        )
        self.model_status_lbl.pack(anchor="w", padx=15, pady=(0, 15))
        
    def start_camera(self):
        self.camera_active = True
        self.video_thread = threading.Thread(target=self.capture_video_loop, daemon=True)
        self.video_thread.start()
        
    def capture_video_loop(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not cap.isOpened():
            self.camera_status_lbl.configure(text="Status: Camera Error", text_color=config.COLORS["danger"])
            self.camera_active = False
            return
            
        # Configure Camera Properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        self.camera_status_lbl.configure(text="Status: Connected", text_color=config.COLORS["success"])
        
        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue
                
            # Flip horizontal for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Process with MediaPipe Hands
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(rgb_frame)
            
            detected_gesture = None
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw skeleton
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmarks list
                raw_landmarks = []
                for lm in hand_landmarks.landmark:
                    raw_landmarks.extend([lm.x, lm.y, lm.z])
                    
                # Normalize landmarks
                normalized = model.preprocess_landmarks(raw_landmarks)
                
                # If Recording Mode is active
                if self.recording_mode:
                    self.recording_buffer.append(normalized.tolist())
                    self.recorded_frames_count += 1
                    
                    # Notify GUI thread of recording progress
                    self.recording_status_queue.put(("PROGRESS", self.recorded_frames_count))
                    
                    if self.recorded_frames_count >= config.RECORDING_FRAMES:
                        self.save_recorded_gesture()
                        
                # Predict gesture if model is available
                elif self.clf:
                    try:
                        probs = self.clf.predict_proba([normalized])[0]
                        max_idx = np.argmax(probs)
                        confidence = probs[max_idx]
                        
                        if confidence >= config.CONFIDENCE_THRESHOLD:
                            label_id = self.clf.classes_[max_idx]
                            detected_gesture = self.label_map[label_id]
                            
                            # Draw prediction on screen
                            cv2.putText(
                                frame,
                                f"{detected_gesture} ({confidence:.0%})",
                                (10, 40),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1.0,
                                (59, 130, 246), # Accent color (blue) in BGR
                                2,
                                cv2.LINE_AA
                            )
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        
            # Format and send frame to GUI queue
            try:
                # Clear queue if full to show only most recent frame
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put((frame, detected_gesture))
            except queue.Full:
                pass
                
            time.sleep(0.015) # Bound to ~60 FPS
            
        cap.release()
        
    def update_gui_loop(self):
        # 1. Update Video Frame
        if not self.frame_queue.empty():
            frame, detected_gesture = self.frame_queue.get()
            
            # Convert OpenCV frame (BGR) to RGB PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Resize image to fit panel
            panel_w = self.camera_panel.winfo_width() - 20
            panel_h = self.camera_panel.winfo_height() - 60
            
            if panel_w > 100 and panel_h > 100:
                # Maintain aspect ratio
                img_w, img_h = pil_img.size
                ratio = min(panel_w/img_w, panel_h/img_h)
                new_w = int(img_w * ratio)
                new_h = int(img_h * ratio)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.video_display.configure(image=img_tk, text="")
            self.video_display.image = img_tk
            
            # Update prediction text
            if detected_gesture:
                self.current_gesture_lbl.configure(
                    text=f"Detected: {detected_gesture}",
                    text_color=config.COLORS["success"]
                )
                self.process_prediction_history(detected_gesture)
            else:
                self.current_gesture_lbl.configure(
                    text="Detected: --",
                    text_color=config.COLORS["accent"]
                )
                
        # 2. Update Recording Progress
        if not self.recording_status_queue.empty():
            status_type, value = self.recording_status_queue.get()
            if status_type == "PROGRESS":
                progress = value / config.RECORDING_FRAMES
                self.record_progress.set(progress)
                self.record_status_lbl.configure(
                    text=f"Recording: Keep holding the sign... ({value}/{config.RECORDING_FRAMES} frames)",
                    text_color=config.COLORS["warning"]
                )
            elif status_type == "DONE":
                self.record_progress.set(0)
                self.record_status_lbl.configure(
                    text=f"Gesture '{value}' recorded successfully! Click 'Train Model' to load it.",
                    text_color=config.COLORS["success"]
                )
                self.recording_mode = False
                self.record_btn.configure(
                    text="🔴 Start Gesture Recording",
                    fg_color=config.COLORS["danger"],
                    hover_color="#DC2626",
                    state="normal"
                )
                self.gesture_entry.configure(state="normal")
                
        self.after(15, self.update_gui_loop)
        
    def process_prediction_history(self, pred):
        self.prediction_history.append(pred)
        if len(self.prediction_history) > self.stable_frames_required:
            self.prediction_history.pop(0)
            
        # Check if predictions are stable
        if len(self.prediction_history) == self.stable_frames_required:
            first = self.prediction_history[0]
            if all(x == first for x in self.prediction_history):
                # We found a stable prediction!
                if first != self.last_translated_word:
                    self.last_translated_word = first
                    
                    # Append to translation console
                    self.history_text.insert("end", f" {first}")
                    self.history_text.see("end")
                    
                    # Async Speak
                    curr_time = time.time()
                    if curr_time - self.last_speech_time > 1.5:  # Rate limit speech to 1.5s
                        self.speak_text(first)
                        self.last_speech_time = curr_time
                        
    def toggle_recording(self):
        if self.recording_mode:
            # Cancel recording
            self.recording_mode = False
            self.record_btn.configure(
                text="🔴 Start Gesture Recording",
                fg_color=config.COLORS["danger"],
                hover_color="#DC2626"
            )
            self.gesture_entry.configure(state="normal")
            self.record_progress.set(0)
            self.record_status_lbl.configure(text="Recording cancelled.", text_color=config.COLORS["text_secondary"])
            return
            
        name = self.gesture_entry.get().strip()
        if not name:
            self.record_status_lbl.configure(text="Error: Enter a gesture name first!", text_color=config.COLORS["danger"])
            return
            
        self.recording_label_name = name
        self.recorded_frames_count = 0
        self.recording_buffer = []
        self.recording_mode = True
        
        self.record_btn.configure(
            text="⏹️ Cancel Recording",
            fg_color="#4B5563",
            hover_color="#374151"
        )
        self.gesture_entry.configure(state="disabled")
        
    def save_recorded_gesture(self):
        # Append data to dataset CSV
        try:
            with open(config.DATASET_PATH, "a") as f:
                for landmarks in self.recording_buffer:
                    line = ",".join([self.recording_label_name] + [f"{val:.6f}" for val in landmarks])
                    f.write(line + "\n")
            
            # Post finished status back to main thread
            self.recording_status_queue.put(("DONE", self.recording_label_name))
        except Exception as e:
            print(f"Error saving gesture: {e}")
            self.recording_mode = False
            
    def train_model_async(self):
        self.train_btn.configure(state="disabled", text="⚡ Training Engine...")
        self.model_status_lbl.configure(text="Model training in progress, please wait...", text_color=config.COLORS["warning"])
        
        def worker():
            try:
                acc, label_map = model.train_gesture_classifier()
                
                # Reload model reference in app
                self.clf, self.label_map = model.load_model()
                
                self.train_btn.configure(state="normal", text="⚙️ Train Translation Model")
                self.model_status_lbl.configure(
                    text=f"Training complete! Accuracy: {acc:.2%}",
                    text_color=config.COLORS["success"]
                )
            except Exception as e:
                self.train_btn.configure(state="normal", text="⚙️ Train Translation Model")
                self.model_status_lbl.configure(
                    text=f"Training Failed: {str(e)[:40]}",
                    text_color=config.COLORS["danger"]
                )
                
        threading.Thread(target=worker, daemon=True).start()
        
    def speak_history(self):
        text = self.history_text.get("1.0", "end").strip()
        if text:
            self.speak_text(text)
            
    def speak_text(self, text):
        def speaker():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 145) # Human-like reading speed
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Text to Speech Error: {e}")
        threading.Thread(target=speaker, daemon=True).start()
        
    def clear_history(self):
        self.history_text.delete("1.0", "end")
        self.last_translated_word = ""
        self.prediction_history.clear()
        
    def on_closing(self):
        self.camera_active = False
        self.destroy()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
