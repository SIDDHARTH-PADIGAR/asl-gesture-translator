import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
import threading

# Streamlit UI
st.title("AI-Based Real-Time Sign Language Detection")

# Load model (cache it to avoid reloading)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("asl_mlp_model.h5")
    label_classes = np.load("label_encoder.npy", allow_pickle=True)
    return model, label_classes

model, label_classes = load_model()

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

# Threading variables
latest_frame = None
predicted_letter = "Waiting..."
lock = threading.Lock()

def process_frame():
    global latest_frame, predicted_letter
    while True:
        if latest_frame is None:
            continue

        # Convert to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(latest_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract hand landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Convert to NumPy array (ensure correct shape)
                landmarks = np.array(landmarks).reshape(1, -1)

                # Predict using model
                prediction = model.predict(landmarks)
                predicted_index = np.argmax(prediction)
                predicted_letter = label_classes[predicted_index]

# Start processing in a separate thread
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame.copy()  # Copy frame for processing thread

    # Display prediction if available
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Prediction: {predicted_letter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame in Streamlit
    frame_placeholder.image(display_frame, channels="BGR", use_container_width=True)

cap.release()
cv2.destroyAllWindows()
