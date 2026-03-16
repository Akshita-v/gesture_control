# Gesture Control (Hand-Gesture Mouse + App Launcher)

A real-time hand-gesture control project built with MediaPipe + Random Forest.
It supports cursor control, click, scrolling, and app launching (Calculator, Chrome, YouTube) from webcam hand gestures.

---

## Features

- Real-time hand landmark detection using **MediaPipe Tasks** (`hand_landmarker.task`)
- Gesture classification using **RandomForestClassifier**
- Gesture stabilization with frame-window majority voting
- Mouse control with smoothing + deadzone filtering
- Action gestures:
  - `0` = Activate system (open palm)
  - `1` = Click (pinch)
  - `2` = Move cursor (index)
  - `3` = Pause system (fist)
  - `4` = Scroll up (two fingers)
  - `5` = Scroll down (three fingers)
  - `6` = Open Calculator
  - `7` = Open Chrome
  - `8` = Open YouTube
- App-launch safety logic to reduce false triggers:
  - cooldown
  - hold-frame requirement
  - confidence threshold requirement
  - re-arm only after non-app frames

---

## Project Structure

- `data_collector.py` → Collect labeled hand landmark samples into CSV
- `train_model_1.py` → Train model, print metrics, save model + encoder
- `mouse_main.py` → Run real-time gesture control app
- `hand_data.csv` → Dataset (label + normalized landmarks)
- `gesture_model_1.pkl` → Trained gesture classifier
- `label_encoder_1.pkl` → Label encoder for gesture IDs
- `hand_landmarker.task` → MediaPipe hand landmark model file

---

## Requirements

- Python 3.9+ (recommended)
- Webcam
- Windows OS (current launcher behavior is Windows-first, e.g. `calc.exe`)

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: `data_collector.py` automatically downloads `hand_landmarker.task` if it is missing.

---

## How to Collect Data

1. Open `data_collector.py`.
2. Set `LABEL_ID` to the gesture you want to record.
3. Run:

```bash
python data_collector.py
```

4. In the camera window:
   - Press `S` to save one sample frame.
   - Press `Q` to quit.
5. Repeat for all gesture labels (`0` to `8`) with balanced sample counts.

### Data Format

Each row in `hand_data.csv`:
- Column 0: gesture label
- Columns 1..N: normalized landmark coordinates (relative to wrist + scaled)

---

## How to Train

Run:

```bash
python train_model_1.py
```

Training script flow:
1. Load CSV
2. Encode labels
3. Stratified train/test split (`test_size=0.2`)
4. Train Random Forest (`n_estimators=300`, `class_weight=balanced_subsample`)
5. Print classification report
6. Retrain on full dataset and save:
   - `gesture_model_1.pkl`
   - `label_encoder_1.pkl`

---

## Performance Report

```text
Training started...

--- PERFORMANCE REPORT ---
Overall Accuracy: 95.24%

Detailed Breakdown per Gesture:
              precision    recall  f1-score   support

           0       0.88      1.00      0.94        15
           1       1.00      0.91      0.95        11
           2       1.00      0.88      0.93         8
           3       1.00      0.80      0.89        10
           4       0.93      1.00      0.97        14
           5       1.00      1.00      1.00        11
           6       0.83      1.00      0.91         5
           7       1.00      1.00      1.00         5
           8       1.00      1.00      1.00         5

    accuracy                           0.95        84
   macro avg       0.96      0.95      0.95        84
weighted avg       0.96      0.95      0.95        84
```

---

## Run the Gesture Controller

```bash
python mouse_main.py
```

Startup message:
- `SYSTEM READY. Show Palm (0) to Start | Fist (3) to Stop.`

Controls:
- Press `q` in the OpenCV window to exit.

Camera window size is set to `960x540` (`CAMERA_WIDTH` / `CAMERA_HEIGHT`). Cursor mapping uses `CURSOR_FRAME_MARGIN = 0.20`, meaning only the central 60% of the frame maps to the full screen for easier edge reach.

---

## Current Runtime Tuning (False Trigger Protection)

In `mouse_main.py`, app gestures use confidence + hold-frame gating:

- Global app cooldown: `APP_COOLDOWN = 2.0`
- Per-gesture hold frames (stabilized frames required before launch):
  - Gesture `6`: `2`
  - Gesture `7`: `3`
  - Gesture `8`: `2`
- Per-gesture minimum confidence (probability of stabilized gesture):
  - Gesture `6`: `0.78`
  - Gesture `7`: `0.80`
  - Gesture `8`: `0.72`
- Per-gesture minimum confidence margin over second-best class:
  - Gesture `6`: `0.10`
  - Gesture `7`: `0.12`
  - Gesture `8`: `0.08`
- Re-arm requirement: at least `6` non-app frames before another app launch is allowed.

Confidence is checked against the stabilized gesture's own probability (not the top class), making detection more reliable when the gesture is held steadily.

---

## Troubleshooting

### 1) App launches accidentally

- Increase hold frames and/or confidence threshold for that gesture in `mouse_main.py`.
- Improve lighting and keep hand fully visible.
- Add more training samples for confusing gesture pairs.
- Add more samples specifically for `6` and `8`; they currently have much lower dataset support than `0`, `4`, and `5`.

### 2) Cursor feels jittery

- Increase `GESTURE_WINDOW` for more stable gesture voting.
- Decrease `SMOOTHING` for smoother movement.
- Increase `CURSOR_DEADZONE` to suppress tiny oscillations.
- Decrease `CURSOR_FRAME_MARGIN` to reduce cursor sensitivity (currently `0.20`).

### 3) Need too much hand movement to reach screen edges

- Increase `CURSOR_FRAME_MARGIN` (e.g. `0.25` or `0.30`) so only the central portion of the camera maps to the full screen — requiring less hand travel.

### 4) Model predicts some gestures poorly

- Collect more balanced samples for low-support classes (`6`, `8` currently have fewer samples).
- Ensure gesture shape consistency during collection.
- Retrain after adding data.

---

## Safety Notes

- `pyautogui.FAILSAFE` is disabled in this project (`False`).
- Keep an easy keyboard path to terminate the app quickly (`q` in the app window).

---

## Quick Start (One Flow)

```bash
# 1) (Optional) collect more samples per gesture
python data_collector.py

# 2) train and save model
python train_model_1.py

# 3) run realtime controller
python mouse_main.py
```

---

## Author Notes

This README reflects the current code behavior in this workspace, including confidence + hold-frame gating for app-launch gestures `6`, `7`, and `8`.
