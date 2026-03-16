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

(Your provided output)

```text
Training started...

--- PERFORMANCE REPORT ---
Overall Accuracy: 93.18%

Detailed Breakdown per Gesture:
              precision    recall  f1-score   support

           0       0.93      0.93      0.93        15
           1       0.92      1.00      0.96        11
           2       0.89      1.00      0.94         8
           3       1.00      0.80      0.89        10
           4       0.93      1.00      0.97        14
           5       0.92      1.00      0.96        11
           6       1.00      0.80      0.89         5
           7       0.89      0.89      0.89         9
           8       1.00      0.80      0.89         5

    accuracy                           0.93        88
   macro avg       0.94      0.91      0.92        88
weighted avg       0.94      0.93      0.93        88
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

---

## Current Runtime Tuning (False Trigger Protection)

In `mouse_main.py`, app gestures use stricter gating:

- Global app cooldown: `APP_COOLDOWN = 2.0`
- Per-gesture hold frames:
  - Gesture `6`: `6`
  - Gesture `7`: `5`
  - Gesture `8`: `3`
- Per-gesture confidence thresholds:
  - Gesture `6`: `0.80`
  - Gesture `7`: `0.75`
  - Gesture `8`: `0.60`
- Re-arm requirement: at least `2` non-app frames before another app launch is allowed.

This helps avoid accidental app opens while transitioning between gestures.

---

## Troubleshooting

### 1) App launches accidentally

- Increase hold frames and/or confidence threshold for that gesture in `mouse_main.py`.
- Improve lighting and keep hand fully visible.
- Add more training samples for confusing gesture pairs.

### 2) Cursor feels jittery

- Increase `GESTURE_WINDOW` for more stable gesture voting.
- Decrease `SMOOTHING` for smoother movement.
- Increase `CURSOR_DEADZONE` to suppress tiny oscillations.

### 3) Model predicts some gestures poorly

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

This README reflects the current code behavior in this workspace, including the latest stricter anti-false-trigger handling for gestures `6` and `7`.
