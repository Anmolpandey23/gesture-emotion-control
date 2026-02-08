# Face Expression + Gesture Control

This project uses your webcam to detect facial emotions and hand gestures in real time. It can also control basic system actions (mouse move, click, scroll, volume, play/pause) and lets you open applications through an on-screen gesture menu.

## Features

- Face emotion detection using a pretrained model.
- Hand gesture recognition using MediaPipe.
- Hand landmark tracking for mouse control.
- Gesture-driven app menu to launch installed applications.
- On-screen help menu showing available gestures.

## Project Files

- inference.py: Main realtime webcam app (emotion + gestures + control).
- data_collection.py: Collects training data for custom models (optional).
- data_training.py: Trains a custom model (optional).
- model.h5 / labels.npy: Older custom model artifacts (optional).
- gesture_recognizer.task: MediaPipe gesture model (auto-downloaded on first run).

## Requirements

- macOS or Windows.
- Python 3.10+.
- Webcam.

Python packages are already installed in the included virtual environment, but you can install them manually if needed:

```
pip install mediapipe opencv-python fer pyautogui certifi
```

## Setup (Recommended)

macOS (included virtual environment):

```
/users/your device name/directory/gesture_control/env/bin/python" inference.py
```

Windows (create and use a venv):

```
python -m venv .venv
.venv\Scripts\activate
pip install mediapipe opencv-python fer pyautogui certifi
python inference.py
```

## macOS Permissions

To allow mouse and keyboard control, enable accessibility permissions:

1) Open System Settings -> Privacy & Security -> Accessibility.
2) Enable your terminal app (Terminal or iTerm) and/or the Python binary.
3) Restart the terminal and run the script again.

Without this, you may see: "osascript is not allowed to send keystrokes".

## Windows Notes

- Volume control uses the optional `pycaw` package when available. If you want system volume control, install:

```
pip install pycaw comtypes
```

- App menu is populated from Start Menu shortcuts.

## How to Run

Start the realtime app:

```
python inference.py
```

Press ESC to exit.

## Gestures (Default)

Mouse + hand landmarks:

- Pointer move: index finger only (other fingers down)
- Click: thumb + index pinch
- Right click: thumb + middle pinch
- Scroll: index + middle extended, move hand up/down

Gesture recognizer actions:

- Volume up: Thumb_Up
- Volume down: Thumb_Down
- Play/Pause: Open_Palm or Closed_Fist

App menu:

- Toggle menu: Pointing_Up
- Move up: Thumb_Up
- Move down: Thumb_Down
- Select/open: Open_Palm
- Exit menu: Closed_Fist

Zoom:

- Zoom in/out: pinch thumb + index while middle finger is extended. Opening the pinch zooms in, closing zooms out.
	- macOS: uses Command +/-
	- Windows: uses Ctrl +/-

## Configuration

Edit the constants near the top of inference.py to tune behavior:

- PROCESS_SCALE: reduce for speed (0.7 is faster, 1.0 is clearer).
- EMOTION_EVERY_N_FRAMES / GESTURE_EVERY_N_FRAMES: higher = less CPU.
- HAND_SMOOTHING / HAND_HOLD_SECONDS: reduce landmark flicker.
- PINCH_THRESHOLD / PINCH_HOLD_FRAMES: adjust click sensitivity.
- MENU_*: change menu gestures or disable the menu.
- GESTURE_HELP_ENABLED: show/hide on-screen help.


## Notes

If you only want pretrained inference, you can ignore data_collection.py and data_training.py. These are for building a custom model.
