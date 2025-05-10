import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
import threading
import random
import time

st.title("AI-Based Real-Time Sign Language Detection")

# Load model and labels
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("asl_mlp_model.h5")
    label_classes = np.load("label_encoder.npy", allow_pickle=True)
    return model, label_classes

model, label_classes = load_model()
label_list = [l for l in label_classes.tolist() if l not in ["J", "Z"]]  # Exclude J and Z

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# OpenCV and UI setup
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
target_placeholder = st.empty()
score_placeholder = st.empty()
performance_placeholder = st.empty()

# Game state
predicted_letter = "Waiting..."
target_sign = random.choice(label_list)
score = 0
total_attempts = 0
correct_attempts = 0
last_correct_time = 0

# Locks
lock = threading.Lock()
latest_frame = None

# Frame processing thread
def process_frame():
    global latest_frame, predicted_letter, score, total_attempts, correct_attempts, target_sign, last_correct_time

    while True:
        if latest_frame is None:
            continue

        frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(latest_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_letter = label_classes[predicted_index]

                # Avoid counting multiple times for same correct gesture
                if time.time() - last_correct_time > 2:  # 2-second cooldown
                    total_attempts += 1
                    if predicted_letter == target_sign:
                        correct_attempts += 1
                        score += 1
                        last_correct_time = time.time()
                        target_sign = random.choice(label_list)

# Start the frame processing thread
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()

# Streamlit UI loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame.copy()

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Prediction: {predicted_letter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Live stats
    accuracy = (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0.0

    target_placeholder.markdown(f"### Show the sign for: `{target_sign}`")
    score_placeholder.markdown(f"### ✅ Score: {score}")
    performance_placeholder.markdown(f"""
    ### 📊 Performance
    - **Total Attempts:** {total_attempts}
    - **Correct Attempts:** {correct_attempts}
    - **Accuracy:** {accuracy:.2f}%
    """)

    frame_placeholder.image(display_frame, channels="BGR", use_container_width=True)

cap.release()
cv2.destroyAllWindows()
