import cv2
import mediapipe as mp
import csv
import numpy as np
import time
import os
import urllib.request

# --- CONFIGURATION ---
LABEL_ID = 7# Change this for each gesture
DATA_FILE = 'hand_data.csv'
MODEL_PATH = "hand_landmarker.task"

# 1. Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

# 2. Setup Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)

detector = HandLandmarker.create_from_options(options)

def normalize_landmarks(landmarks):
    temp_list = []
    # Wrist is the base (index 0)
    base_x, base_y = landmarks[0].x, landmarks[0].y
    
    for lm in landmarks:
        temp_list.append([lm.x - base_x, lm.y - base_y])

    flat_list = np.array(temp_list).flatten()
    max_val = np.abs(flat_list).max()
    if max_val != 0:
        flat_list = flat_list / max_val
    return flat_list.tolist()

cap = cv2.VideoCapture(0)
print(f"Recording for Label: {LABEL_ID}. Press 'S' to save, 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Tasks API requires a timestamp in milliseconds
    timestamp = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp)

    if results.hand_landmarks:
        # Get the first hand
        hand_lms = results.hand_landmarks[0]
        
        # Draw for visual feedback
        h, w, _ = frame.shape
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            normalized_data = [LABEL_ID] + normalize_landmarks(hand_lms)
            with open(DATA_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(normalized_data)
            print(f"Saved Frame for Label {LABEL_ID}")
        elif key == ord('q'):
            break

    cv2.imshow("Tasks API - Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
detector.close()