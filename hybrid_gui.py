
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from tensorflow.keras.models import load_model
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector

# Load both models
lstm_model = load_model("my_modelupg.h5")  # LSTM model
cnn_model = load_model("cnn8grps_rad1_model.h5")  # CNN model

# Gesture classes for LSTM
gesture_classes = ['None', 'Hello', 'Bye', 'Thank You', 'My name', 'Yes', 'No', 'How are you', 'I am good']

# Setup pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 120)

# MediaPipe for landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_mp = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# OpenCV HandDetector for CNN input
hand_detector = HandDetector(maxHands=1)
offset = 30

# Frame buffer for LSTM input
frame_buffer = deque(maxlen=10)

# GUI setup
class HybridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator (Hybrid CNN + LSTM)")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Prediction:", font=("Helvetica", 24))
        self.label.pack()

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.prediction = "None"
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def extract_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_mp.process(frame_rgb)
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        while len(keypoints) < 126:
            keypoints.append(0)
        return np.array(keypoints) if len(keypoints) == 126 else None

    def predict_lstm(self):
        if len(frame_buffer) == 10:
            input_data = np.array(frame_buffer).reshape(1, 10, 126)
            probs = lstm_model.predict(input_data, verbose=0)
            confidence = np.max(probs)
            pred = gesture_classes[np.argmax(probs)]
            return pred if confidence > 0.8 else "None"
        return "None"

    def predict_cnn(self, frame):
        hands, img = hand_detector.findHands(frame, draw=False)
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            cropped = frame[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            try:
                cropped = cv2.resize(cropped, (400, 400))
                white[:cropped.shape[0], :cropped.shape[1]] = cropped
                input_img = white.reshape(1, 400, 400, 3)
                pred_idx = np.argmax(cnn_model.predict(input_img, verbose=0))
                return chr(65 + pred_idx)  # A-Z
            except:
                return "None"
        return "None"

    def speak_prediction(self, prediction):
        if prediction != "None":
            engine.say(prediction)
            engine.runAndWait()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        keypoints = self.extract_landmarks(frame)
        if keypoints is not None:
            frame_buffer.append(keypoints)
        lstm_pred = self.predict_lstm()
        cnn_pred = self.predict_cnn(frame)

        # Decision logic
        final_pred = lstm_pred if lstm_pred != "None" else cnn_pred

        if final_pred != self.prediction:
            self.prediction = final_pred
            self.label.config(text=f"Prediction: {self.prediction}")
            self.speak_prediction(final_pred)

        # Display video
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_close(self):
        self.cap.release()
        self.root.destroy()

# Start GUI
root = tk.Tk()
app = HybridApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_close)
root.mainloop()
