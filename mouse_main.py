import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import subprocess
import time
import os
import webbrowser
from collections import Counter, deque
from mediapipe.tasks.python.vision import hand_landmarker

# --- 1. CONFIGURATION ---
MODEL_PATH = "gesture_model_1.pkl"
ENCODER_PATH = "label_encoder_1.pkl"
TASK_FILE = "hand_landmarker.task"

pyautogui.FAILSAFE = False
SMOOTHING = 0.16            # Higher = less smoothing, lower = smoother cursor movement
CURSOR_DEADZONE = 10        # Ignore tiny hand shifts before moving the cursor
CURSOR_FRAME_MARGIN = 0.20  # Smaller active zone = less hand movement needed to reach screen edges
GESTURE_WINDOW = 6          # More frames = steadier gesture stabilization
SCROLL_COOLDOWN = 0.12      # Seconds between scroll events
APP_COOLDOWN = 2.0          # Seconds before the same app can be launched again
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
WINDOW_NAME = "Hand-Gesture Mouse Controller"
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540
APP_HOLD_FRAMES = 3         # Require same app gesture for N stabilized frames
APP_MIN_CONFIDENCE_MAP = {
    "6": 0.78,
    "7": 0.80,
    "8": 0.72,
}
APP_MIN_MARGIN_MAP = {
    "6": 0.10,
    "7": 0.12,
    "8": 0.08,
}
APP_HOLD_FRAMES_MAP = {
    "6": 2,
    "7": 3,
    "8": 2,
}
APP_REARM_NON_APP_FRAMES = 6
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
]

# --- 2. INITIALIZE AI & TOOLS ---
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
CLASS_INDEX_MAP = {str(label).strip(): idx for idx, label in enumerate(encoder.classes_)}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_FILE),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6
)
detector = HandLandmarker.create_from_options(options)

def launch_chrome(url=None):
    chrome_exe = next((path for path in CHROME_PATHS if os.path.exists(path)), None)

    if chrome_exe:
        cmd = [chrome_exe]
        if url:
            cmd.append(url)
        subprocess.Popen(cmd)
        return

    if url:
        try:
            os.startfile(url)
        except OSError:
            webbrowser.open(url)
    else:
        try:
            os.startfile("https://www.google.com")
        except OSError:
            webbrowser.open("https://www.google.com")

def launch_calculator():
    # Try UWP protocol (most reliable on Windows 10/11)
    try:
        os.startfile("calculator:")
        return
    except Exception as e:
        pass

    # Fallback: calc.exe via cmd
    try:
        subprocess.Popen(["cmd", "/c", "start", "calc"])
        return
    except Exception as e:
        pass

    try:
        subprocess.Popen(["calc.exe"])
    except Exception as e:
        pass

def launch_youtube():
    try:
       os.startfile("https://www.youtube.com")
    except OSError:
            webbrowser.open("https://www.youtube.com")
def draw_hand_landmarks(frame, landmarks):
    h, w, _ = frame.shape

    # Draw hand skeleton lines
    for connection in HAND_CONNECTIONS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Draw landmark dots (outlined for better visibility)
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), -1)      # outer black ring
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)    # inner green dot

def get_features(landmarks):
    base_x, base_y = landmarks[0].x, landmarks[0].y
    temp = [[lm.x - base_x, lm.y - base_y] for lm in landmarks]
    flat = np.array(temp).flatten()
    max_val = np.abs(flat).max()
    return (flat / max_val).reshape(1, -1) if max_val != 0 else flat.reshape(1, -1)

def draw_horizontal_legend(frame, active_gesture):
    rows = [
        [
            ("0", "0:Palm-Activate"),
            ("1", "1:Pinch-Click"),
            ("2", "2:Index-Move"),
        ],
        [
            ("3", "3:Fist-Pause"),
            ("4", "4:TwoF-ScrollUp"),
            ("5", "5:ThreeF-ScrollDown"),
        ],
        [
            ("6", "6:Calculator"),
            ("7", "7:Chrome"),
            ("8", "8:YouTube"),
        ],
    ]

    start_x = 10
    start_y = 72
    row_gap = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for row_idx, row in enumerate(rows):
        x = start_x
        y = start_y + row_idx * row_gap

        for item_idx, (gid, label) in enumerate(row):
            color = (0, 255, 255) if active_gesture == gid else (180, 180, 0)
            cv2.putText(frame, label, (x, y), font, font_scale, color, thickness)
            text_width = cv2.getTextSize(label, font, font_scale, thickness)[0][0]
            x += text_width

            if item_idx < len(row) - 1:
                sep = " | "
                cv2.putText(frame, sep, (x, y), font, font_scale, (180, 180, 0), thickness)
                sep_width = cv2.getTextSize(sep, font, font_scale, thickness)[0][0]
                x += sep_width

# --- 3. SYSTEM STATE ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
sw, sh = pyautogui.size()
is_active = False
last_scroll_time = 0
last_app_time = 0
smooth_x = None
smooth_y = None
gesture_history = deque(maxlen=GESTURE_WINDOW)
prev_gesture = None
app_ready_prev = False
app_non_gesture_count = 0
app_armed = True
app_hold_gesture = None
app_hold_count = 0

GESTURE_LABELS = {
    "0": "Palm -> Activate",
    "1": "Pinch -> Click",
    "2": "Index -> Move Cursor",
    "3": "Fist -> Pause",
    "4": "Two Fingers -> Scroll Up",
    "5": "Three Fingers -> Scroll Down",
    "6": "Open Calculator",
    "7": "Open Chrome",
    "8": "Open YouTube",
}

print("SYSTEM READY. Show Palm (0) to Start | Fist (3) to Stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process hand landmarks
    res = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if res.hand_landmarks:
        lms = res.hand_landmarks[0]

        # A. PREDICT GESTURE
        feat = get_features(lms)
        probs = model.predict_proba(feat)[0]
        max_prob_idx = int(np.argmax(probs))
        max_prob = float(probs[max_prob_idx])
        sorted_prob_indices = np.argsort(probs)
        second_prob = float(probs[sorted_prob_indices[-2]]) if len(sorted_prob_indices) > 1 else 0.0
        pred = model.predict(feat)
        raw_gesture = str(encoder.inverse_transform(pred)[0]).strip()

        # Stabilize gesture by majority vote across recent frames.
        gesture_history.append(raw_gesture)
        gesture = Counter(gesture_history).most_common(1)[0][0]

        # B. TRACKING (Index Tip = Landmark 8)
        itip = lms[8]
        clipped_x = np.clip(itip.x, CURSOR_FRAME_MARGIN, 1 - CURSOR_FRAME_MARGIN)
        clipped_y = np.clip(itip.y, CURSOR_FRAME_MARGIN, 1 - CURSOR_FRAME_MARGIN)
        target_x = np.interp(clipped_x, [CURSOR_FRAME_MARGIN, 1 - CURSOR_FRAME_MARGIN], [0, sw])
        target_y = np.interp(clipped_y, [CURSOR_FRAME_MARGIN, 1 - CURSOR_FRAME_MARGIN], [0, sh])

        # C. ACTIVATION LOGIC
        if gesture == "0":   # OPEN PALM
            is_active = True
        elif gesture == "3": # FIST
            is_active = False
            smooth_x, smooth_y = None, None

        # D. WORKER LOGIC
        if is_active:
            # 1. CURSOR CONTROL (ID 2 - Index Finger)
            if gesture == "2":
                if smooth_x is None or smooth_y is None:
                    smooth_x, smooth_y = target_x, target_y
                else:
                    smooth_x += (target_x - smooth_x) * SMOOTHING
                    smooth_y += (target_y - smooth_y) * SMOOTHING

                curr_x, curr_y = pyautogui.position()
                if abs(smooth_x - curr_x) > CURSOR_DEADZONE or abs(smooth_y - curr_y) > CURSOR_DEADZONE:
                    pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
            else:
                smooth_x, smooth_y = None, None

            # 2. CLICK / SELECT (ID 1 - Pinch, edge-triggered)
            if gesture == "1" and prev_gesture != "1":
                pyautogui.click()

            # 3. SCROLLING (ID 4 - Up | ID 5 - Down)
            now = time.time()
            if now - last_scroll_time > SCROLL_COOLDOWN:
                if gesture == "4":   # Two Fingers
                    pyautogui.scroll(40)
                    last_scroll_time = now
                elif gesture == "5": # Three Fingers
                    pyautogui.scroll(-40)
                    last_scroll_time = now

            cv2.putText(frame, f"STATUS: ACTIVE | GESTURE: {gesture}", (10, 40), 1, 1.3, (0, 255, 0), 2)
        else:
            smooth_x, smooth_y = None, None
            cv2.putText(frame, "STATUS: PAUSED (Show Palm)", (10, 40), 1, 1.3, (0, 0, 255), 2)

        # 4. APP LAUNCHER (IDs 6/7/8, based on stabilized gesture)
        app_candidate = None
        if gesture in {"6", "7", "8"}:
            gesture_idx = CLASS_INDEX_MAP.get(gesture)
            if gesture_idx is not None:
                gesture_prob = float(probs[gesture_idx])
                confidence_margin = gesture_prob - second_prob
                min_conf = APP_MIN_CONFIDENCE_MAP.get(gesture, 0.75)
                min_margin = APP_MIN_MARGIN_MAP.get(gesture, 0.08)
                if gesture_prob >= min_conf and confidence_margin >= min_margin:
                    app_candidate = gesture

        if app_candidate == app_hold_gesture:
            app_hold_count += 1
        else:
            app_hold_gesture = app_candidate
            app_hold_count = 1 if app_candidate else 0

        if app_candidate is None:
            app_non_gesture_count += 1
            if app_non_gesture_count >= APP_REARM_NON_APP_FRAMES:
                app_armed = True
        else:
            app_non_gesture_count = 0

        hold_required = APP_HOLD_FRAMES_MAP.get(app_candidate, APP_HOLD_FRAMES) if app_candidate else APP_HOLD_FRAMES
        app_ready = app_candidate is not None and app_hold_count >= hold_required
        now = time.time()
        if app_ready and not app_ready_prev and app_armed and now - last_app_time > APP_COOLDOWN:
            if app_candidate == "6":   # Calculator
                launch_calculator()
                last_app_time = now
                app_armed = False
            elif app_candidate == "7": # Chrome
                launch_chrome()
                last_app_time = now
                app_armed = False
            elif app_candidate == "8": # YouTube
                launch_youtube()
                last_app_time = now
                app_armed = False
        app_ready_prev = app_ready

        # Show gesture names in horizontal format with active gesture highlight.
        draw_horizontal_legend(frame, gesture)
        cv2.putText(frame, f"CURRENT: {gesture}", (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        prev_gesture = gesture

        # Draw hand landmarks as dots + lines
        draw_hand_landmarks(frame, lms)

    else:
        smooth_x, smooth_y = None, None
        app_ready_prev = False
        app_non_gesture_count = 0
        app_armed = True
        app_hold_gesture = None
        app_hold_count = 0

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

detector.close()
cap.release()
cv2.destroyAllWindows()