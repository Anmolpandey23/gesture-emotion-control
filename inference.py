import importlib
import os
import ssl
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List

import certifi

import cv2
import mediapipe as mp
import pyautogui
from fer.fer import FER
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


GESTURE_MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
	"gesture_recognizer/float16/1/gesture_recognizer.task"
)
GESTURE_MODEL_PATH = "gesture_recognizer.task"


def ensure_gesture_model(path: str, url: str) -> str:
	if os.path.exists(path):
		return path

	print("Downloading gesture model...")
	ssl_context = ssl.create_default_context(cafile=certifi.where())
	with urllib.request.urlopen(url, context=ssl_context) as response, open(path, "wb") as out_file:
		out_file.write(response.read())
	return path


def create_gesture_recognizer(model_path: str):
	base_options = python.BaseOptions(model_asset_path=model_path)
	options = vision.GestureRecognizerOptions(
		base_options=base_options,
		running_mode=vision.RunningMode.VIDEO,
	)
	return vision.GestureRecognizer.create_from_options(options)


gesture_model_path = ensure_gesture_model(GESTURE_MODEL_PATH, GESTURE_MODEL_URL)
gesture_recognizer = create_gesture_recognizer(gesture_model_path)
emotion_detector = FER()
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
hands_detector = hands.Hands(
	max_num_hands=2,
	model_complexity=1,
	min_detection_confidence=0.7,
	min_tracking_confidence=0.8,
)

# Action settings (edit these to your preferences).
REQUIRE_CONFIRMATION = False
COOLDOWN_SECONDS = 2.0
CLICK_COOLDOWN_SECONDS = 0.8
RIGHT_CLICK_COOLDOWN_SECONDS = 1.0
SCROLL_COOLDOWN_SECONDS = 0.1
PINCH_THRESHOLD = 0.05
PINCH_HOLD_FRAMES = 3
ZOOM_ENABLED = True
ZOOM_DELTA_THRESHOLD = 0.006
ZOOM_COOLDOWN_SECONDS = 0.4
SCROLL_SENSITIVITY = 1200
SCROLL_DEADZONE = 0.004
MOUSE_CONTROL_ENABLED = True
PROCESS_SCALE = 0.7
PROCESS_EVERY_N_FRAMES = 1
ENABLE_EMOTION = True
EMOTION_EVERY_N_FRAMES = 6
MENU_ENABLED = True
MENU_TOGGLE_GESTURE = "Pointing_Up"
MENU_UP_GESTURE = "Thumb_Up"
MENU_DOWN_GESTURE = "Thumb_Down"
MENU_SELECT_GESTURE = "Open_Palm"
MENU_EXIT_GESTURE = "Closed_Fist"
MENU_COOLDOWN_SECONDS = 0.6
MENU_MAX_ITEMS = 8
MOUSE_SMOOTHING = 0.25
MOUSE_DEADZONE = 0.003
HAND_HOLD_SECONDS = 1.5
HAND_SMOOTHING = 0.6
GESTURE_HELP_ENABLED = True
GESTURE_EVERY_N_FRAMES = 6
GESTURE_STALE_SECONDS = 0.5
IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
ZOOM_MODIFIER = "command" if IS_MAC else "ctrl"

ACTION_GESTURES = {
	"Thumb_Up": "volume_up",
	"Thumb_Down": "volume_down",
	"Open_Palm": "play_pause",
	"Closed_Fist": "play_pause",
}

cap = cv2.VideoCapture(0)

last_action_time = 0.0
last_click_time = 0.0
last_right_click_time = 0.0
last_scroll_time = 0.0
previous_scroll_y = None
mouse_x = None
mouse_y = None
frame_count = 0
pinch_frames = 0
right_pinch_frames = 0
pinch_active = False
right_pinch_active = False
last_pinch_distance = None
last_zoom_time = 0.0
menu_open = False
menu_index = 0
menu_last_nav = 0.0
last_hands_time = {"Left": 0.0, "Right": 0.0}
last_smoothed_hands = {"Left": None, "Right": None}
APP_PATHS: Dict[str, str] = {}


def list_applications() -> List[str]:
	apps: List[str] = []
	if IS_WINDOWS:
		start_menu_paths = [
			os.path.join(os.environ.get("ProgramData", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
			os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
		]
		for base in start_menu_paths:
			if not os.path.isdir(base):
				continue
			for root, _, files in os.walk(base):
				for name in files:
					if name.endswith(".lnk"):
						display = name[:-4]
						path = os.path.join(root, name)
						APP_PATHS[display] = path
						apps.append(display)
	else:
		for base in ["/Applications", os.path.expanduser("~/Applications")]:
			if not os.path.isdir(base):
				continue
			for name in os.listdir(base):
				if name.endswith(".app"):
					apps.append(name[:-4])
	apps = sorted(set(apps), key=str.lower)
	return apps


APPS = list_applications()

last_emotion_text = "Disabled"
last_gesture_text = "No gesture"
last_action_text = ""
last_top_gesture_name = None
last_gesture_time = 0.0


def clamp_volume(value: int) -> int:
	return max(0, min(100, value))


def set_volume(delta: int) -> None:
	if IS_MAC:
		# macOS volume control via AppleScript.
		get_script = "output volume of (get volume settings)"
		result = subprocess.run(["osascript", "-e", get_script], capture_output=True, text=True, check=False)
		try:
			current = int(result.stdout.strip())
		except ValueError:
			current = 50
		new_volume = clamp_volume(current + delta)
		set_script = f"set volume output volume {new_volume}"
		subprocess.run(["osascript", "-e", set_script], check=False)
		return

	if IS_WINDOWS:
		try:
			comtypes = importlib.import_module("comtypes")
			pycaw = importlib.import_module("pycaw.pycaw")
			clsctx_all = getattr(comtypes, "CLSCTX_ALL")
			audio_utils = getattr(pycaw, "AudioUtilities")
			endpoint = getattr(pycaw, "IAudioEndpointVolume")
			devices = audio_utils.GetSpeakers()
			interface = devices.Activate(endpoint._iid_, clsctx_all, None)
			volume = interface.QueryInterface(endpoint)
			current = int(volume.GetMasterVolumeLevelScalar() * 100)
			new_volume = clamp_volume(current + delta)
			volume.SetMasterVolumeLevelScalar(new_volume / 100.0, None)
			return
		except Exception:
			key = "volumeup" if delta > 0 else "volumedown"
			pyautogui.press(key)
			return


def activate_chrome() -> None:
	if IS_MAC:
		subprocess.run(["osascript", "-e", 'tell application "Google Chrome" to activate'], check=False)


def play_pause_youtube() -> None:
	if IS_MAC:
		activate_chrome()
		# Space toggles play/pause in YouTube when the player is focused.
		subprocess.run(["osascript", "-e", 'tell application "System Events" to keystroke " "'], check=False)
		return

	if IS_WINDOWS:
		pyautogui.press("playpause")
		return


def open_app(app_name: str) -> None:
	if IS_MAC:
		subprocess.run(["open", "-a", app_name], check=False)
		return
	if IS_WINDOWS:
		path = APP_PATHS.get(app_name, app_name)
		try:
			os.startfile(path)
		except OSError:
			pass


def perform_action(action: str) -> None:
	if action == "volume_up":
		set_volume(6)
		return
	if action == "volume_down":
		set_volume(-6)
		return
	if action == "play_pause":
		play_pause_youtube()
		return


def is_extended(landmarks, tip_id: int, pip_id: int) -> bool:
	return landmarks[tip_id].y < landmarks[pip_id].y


def distance(landmarks, a: int, b: int) -> float:
	dx = landmarks[a].x - landmarks[b].x
	dy = landmarks[a].y - landmarks[b].y
	return (dx * dx + dy * dy) ** 0.5


def smooth_landmarks(current_landmarks, previous_landmarks, alpha: float):
	if previous_landmarks is None:
		return current_landmarks
	for i, point in enumerate(current_landmarks):
		prev = previous_landmarks[i]
		point.x = (1 - alpha) * prev.x + alpha * point.x
		point.y = (1 - alpha) * prev.y + alpha * point.y
		point.z = (1 - alpha) * prev.z + alpha * point.z
	return current_landmarks


def draw_overlay(frame, emotion_text: str, gesture_text: str, action_text: str) -> None:
	cv2.putText(frame, f"Emotion: {emotion_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
	cv2.putText(frame, f"Gesture: {gesture_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
	if action_text:
		cv2.putText(frame, action_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)

	if MENU_ENABLED:
		menu_title = "App Menu" if menu_open else "Menu: Pointing_Up"
		cv2.putText(frame, menu_title, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		if menu_open:
			if not APPS:
				cv2.putText(frame, "No apps found", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
			else:
				start = max(0, menu_index - (MENU_MAX_ITEMS // 2))
				end = min(len(APPS), start + MENU_MAX_ITEMS)
				for idx, app_name in enumerate(APPS[start:end], start=start):
					y = 200 + (idx - start) * 24
					color = (0, 255, 255) if idx == menu_index else (200, 200, 200)
					prefix = ">" if idx == menu_index else " "
					cv2.putText(frame, f"{prefix} {app_name}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

	if GESTURE_HELP_ENABLED:
		help_x = 20
		help_y = frame.shape[0] - 180
		cv2.putText(frame, "Gestures:", (help_x, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
		lines = [
			"Pointer: index only",
			"Click: thumb + index pinch",
			"Right click: thumb + middle pinch",
			"Scroll: index + middle up/down",
			"Zoom: pinch + middle extended",
			"Volume: Thumb_Up / Thumb_Down",
			"Play/Pause: Open_Palm or Closed_Fist",
			"Menu: Pointing_Up (toggle)",
			"Menu select: Open_Palm",
		]
		for i, line in enumerate(lines):
			cv2.putText(
				frame,
				line,
				(help_x, help_y + 22 + i * 20),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(180, 255, 180),
				1,
			)
while True:
	ok, frame = cap.read()
	if not ok:
		break

	frame = cv2.flip(frame, 1)
	frame_count += 1
	if PROCESS_EVERY_N_FRAMES > 1 and (frame_count % PROCESS_EVERY_N_FRAMES) != 0:
		draw_overlay(frame, last_emotion_text, last_gesture_text, last_action_text)
		cv2.imshow("window", frame)
		if cv2.waitKey(1) == 27:
			cv2.destroyAllWindows()
			cap.release()
			break
		continue

	if PROCESS_SCALE != 1.0:
		small = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
	else:
		small = frame
	rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

	# Face expression detection (pretrained).
	emotion_text = last_emotion_text
	if ENABLE_EMOTION and (frame_count % EMOTION_EVERY_N_FRAMES == 0):
		emotion = emotion_detector.top_emotion(rgb)
		emotion_text = "No face"
		if emotion and emotion[0]:
			if emotion[1] is None:
				emotion_text = emotion[0]
			else:
				emotion_text = f"{emotion[0]} ({emotion[1]:.2f})"

	# Hand gesture detection (pretrained).
	gesture_text = last_gesture_text
	top_gesture_name = last_top_gesture_name
	if frame_count % GESTURE_EVERY_N_FRAMES == 0:
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		timestamp_ms = int(time.time() * 1000)
		result = gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)
		gesture_text = "No gesture"
		top_gesture_name = None
		if result.gestures:
			top_gesture = result.gestures[0][0]
			top_gesture_name = top_gesture.category_name
			gesture_text = f"{top_gesture_name} ({top_gesture.score:.2f})"
		last_gesture_time = time.time()

	if last_top_gesture_name and (time.time() - last_gesture_time) > GESTURE_STALE_SECONDS:
		top_gesture_name = None
		gesture_text = "No gesture"

	now = time.time()
	action_text = ""
	if not menu_open and top_gesture_name in ACTION_GESTURES:
		action = ACTION_GESTURES[top_gesture_name]
		if REQUIRE_CONFIRMATION:
			action_text = f"Pending: {action}"
		elif (now - last_action_time) >= COOLDOWN_SECONDS:
			perform_action(action)
			last_action_time = now
			action_text = f"Action: {action}"

	# App menu navigation with gestures.
	if MENU_ENABLED and top_gesture_name:
		if top_gesture_name == MENU_TOGGLE_GESTURE and (now - menu_last_nav) >= MENU_COOLDOWN_SECONDS:
			menu_open = not menu_open
			menu_last_nav = now
		elif menu_open and APPS:
			if top_gesture_name == MENU_UP_GESTURE and (now - menu_last_nav) >= MENU_COOLDOWN_SECONDS:
				menu_index = (menu_index - 1) % len(APPS)
				menu_last_nav = now
			elif top_gesture_name == MENU_DOWN_GESTURE and (now - menu_last_nav) >= MENU_COOLDOWN_SECONDS:
				menu_index = (menu_index + 1) % len(APPS)
				menu_last_nav = now
			elif top_gesture_name == MENU_SELECT_GESTURE and (now - menu_last_nav) >= MENU_COOLDOWN_SECONDS:
				open_app(APPS[menu_index])
				menu_last_nav = now
			elif top_gesture_name == MENU_EXIT_GESTURE and (now - menu_last_nav) >= MENU_COOLDOWN_SECONDS:
				menu_open = False
				menu_last_nav = now


	# Hand landmarks for mouse-style gestures (prefer Right hand, else Left).
	hands_result = hands_detector.process(rgb)
	if hands_result.multi_hand_landmarks:
		for hand_index, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
			label = "Hand"
			if hands_result.multi_handedness and hand_index < len(hands_result.multi_handedness):
				label = hands_result.multi_handedness[hand_index].classification[0].label
			prev = last_smoothed_hands.get(label)
			prev_points = prev.landmark if prev else None
			smooth_landmarks(hand_landmarks.landmark, prev_points, HAND_SMOOTHING)
			last_smoothed_hands[label] = hand_landmarks
			last_hands_time[label] = now

	control_hand = last_smoothed_hands["Right"] or last_smoothed_hands["Left"]
	if control_hand:
		landmarks = control_hand.landmark
		pinch_distance = distance(landmarks, 4, 8)
		right_pinch_distance = distance(landmarks, 4, 12)
		index_extended = is_extended(landmarks, 8, 6)
		middle_extended = is_extended(landmarks, 12, 10)
		ring_extended = is_extended(landmarks, 16, 14)
		pinky_extended = is_extended(landmarks, 20, 18)

		if ZOOM_ENABLED and middle_extended and not ring_extended and not pinky_extended:
			if last_pinch_distance is not None:
				delta = pinch_distance - last_pinch_distance
				if abs(delta) >= ZOOM_DELTA_THRESHOLD and (now - last_zoom_time) >= ZOOM_COOLDOWN_SECONDS:
					if delta > 0:
						pyautogui.hotkey(ZOOM_MODIFIER, "+")
						action_text = "Action: zoom in"
					else:
						pyautogui.hotkey(ZOOM_MODIFIER, "-")
						action_text = "Action: zoom out"
					last_zoom_time = now
			last_pinch_distance = pinch_distance
		else:
			last_pinch_distance = None

		if MOUSE_CONTROL_ENABLED and index_extended and not middle_extended and not ring_extended and not pinky_extended:
			screen_w, screen_h = pyautogui.size()
			ix = landmarks[8].x
			iy = landmarks[8].y
			if mouse_x is None or mouse_y is None:
				mouse_x = ix
				mouse_y = iy
			if abs(ix - mouse_x) > MOUSE_DEADZONE or abs(iy - mouse_y) > MOUSE_DEADZONE:
				mouse_x = (1 - MOUSE_SMOOTHING) * mouse_x + MOUSE_SMOOTHING * ix
				mouse_y = (1 - MOUSE_SMOOTHING) * mouse_y + MOUSE_SMOOTHING * iy
				pyautogui.moveTo(int(mouse_x * screen_w), int(mouse_y * screen_h))

		pinch_frames = pinch_frames + 1 if pinch_distance < PINCH_THRESHOLD else 0
		right_pinch_frames = right_pinch_frames + 1 if right_pinch_distance < PINCH_THRESHOLD else 0

		if not pinch_active and pinch_frames >= PINCH_HOLD_FRAMES and (now - last_click_time) >= CLICK_COOLDOWN_SECONDS and not middle_extended:
			pyautogui.click()
			last_click_time = now
			pinch_active = True
			action_text = "Action: click"
		elif not right_pinch_active and right_pinch_frames >= PINCH_HOLD_FRAMES and (now - last_right_click_time) >= RIGHT_CLICK_COOLDOWN_SECONDS:
			pyautogui.rightClick()
			last_right_click_time = now
			right_pinch_active = True
			action_text = "Action: right click"
		else:
			if index_extended and middle_extended:
				current_scroll_y = (landmarks[8].y + landmarks[12].y) / 2.0
				if previous_scroll_y is not None:
					delta = current_scroll_y - previous_scroll_y
					if abs(delta) > SCROLL_DEADZONE and (now - last_scroll_time) >= SCROLL_COOLDOWN_SECONDS:
						scroll_amount = int(-delta * SCROLL_SENSITIVITY)
						if scroll_amount != 0:
							pyautogui.scroll(scroll_amount)
							last_scroll_time = now
							action_text = "Action: scroll"
				previous_scroll_y = current_scroll_y
			else:
				previous_scroll_y = None

		if pinch_distance >= PINCH_THRESHOLD:
			pinch_active = False
		if right_pinch_distance >= PINCH_THRESHOLD:
			right_pinch_active = False

	# Draw landmarks and label left/right hands (use cached data briefly).
	for label in ("Left", "Right"):
		hand_landmarks = last_smoothed_hands.get(label)
		if hand_landmarks and (now - last_hands_time[label]) <= HAND_HOLD_SECONDS:
			drawing.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)
			wrist = hand_landmarks.landmark[0]
			cx = int(wrist.x * frame.shape[1])
			cy = int(wrist.y * frame.shape[0])
			cv2.putText(frame, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

	last_emotion_text = emotion_text
	last_gesture_text = gesture_text
	last_action_text = action_text
	draw_overlay(frame, emotion_text, gesture_text, action_text)

	cv2.imshow("window", frame)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break