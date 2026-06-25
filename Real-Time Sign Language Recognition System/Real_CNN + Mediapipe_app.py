# ============================================================
#  app.py — Flask ASL Recognition with MediaPipe + Real CNN
#  FYP - Real-Time Sign Language Recognition System
#
#  Model input : Cropped hand image (64 x 64 x 3)
#  Model type  : CNN
#  MediaPipe   : Used for hand detection and cropping
#
#  Works with:
#  - Real_cnn_mediapipe_image_model.keras
#  - Real_cnn_mediapipe_image_labels.json
# ============================================================

import argparse
import json
import os
import time
import urllib.request
from collections import Counter, deque
from threading import Lock

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template, request


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE = 64
HAND_LANDMARKER_PATH = "hand_landmarker.task"


# ─────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Flask ASL Recognition with MediaPipe + CNN"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="Real_cnn_mediapipe_image_model.keras"
    )

    parser.add_argument(
        "--labels_path",
        type=str,
        default="Real_cnn_mediapipe_image_labels.json"
    )

    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--min_conf", type=float, default=0.70)
    parser.add_argument("--stable_frames", type=int, default=8)
    parser.add_argument("--cooldown_sec", type=float, default=0.7)

    return parser.parse_args()


# ─────────────────────────────────────────────
# WORD SUGGESTION MODULE
# ─────────────────────────────────────────────
WORDS_FILE = "english_words.txt"

def load_words():
    if not os.path.exists(WORDS_FILE):
        return [
            "apple", "apply", "about", "after", "again",
            "ask", "hello", "help", "please", "thanks"
        ]

    with open(WORDS_FILE, "r", encoding="utf-8") as f:
        return [w.strip().lower() for w in f if w.strip().isalpha()]


WORDS = load_words()


def get_suggestions(sentence, limit=20):
    if sentence.endswith(" "):
        return []

    sentence = sentence.lower().strip()

    if not sentence:
        return []

    last_word = sentence.split()[-1]

    if not last_word:
        return []

    matches = []

    for word in WORDS:
        if word.startswith(last_word):
            matches.append(word)

            if len(matches) == limit:
                break

    return matches


# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
def create_hand_landmarker():
    """
    Create MediaPipe HandLandmarker using MediaPipe Tasks API.
    This avoids the mp.solutions.hands error.
    """
    base_options = mp.tasks.BaseOptions(
        model_asset_path=HAND_LANDMARKER_PATH
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# CROP HAND FOR CNN INPUT
# ─────────────────────────────────────────────
def crop_hand_from_result(frame, result):
    """
    Crop hand region using MediaPipe landmark result.
    Returns CNN-ready image with shape (1, 64, 64, 3).
    """

    if not result or not result.hand_landmarks:
        return None

    h, w = frame.shape[:2]
    landmarks = result.hand_landmarks[0]

    x_coords = []
    y_coords = []

    for lm in landmarks:
        x_coords.append(int(lm.x * w))
        y_coords.append(int(lm.y * h))

    padding = 30

    x_min = max(min(x_coords) - padding, 0)
    y_min = max(min(y_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, w)
    y_max = min(max(y_coords) + padding, h)

    hand_crop = frame[y_min:y_max, x_min:x_max]

    if hand_crop.size == 0:
        return None

    hand_crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
    hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)

    hand_input = np.expand_dims(hand_crop, axis=0).astype(np.float32)

    return hand_input


# ─────────────────────────────────────────────
# DRAW LANDMARKS ON FRAME
# ─────────────────────────────────────────────
def draw_landmarks_on_frame(frame, landmarks_result):
    """
    Draw 21 hand landmarks and connections on the frame.
    """

    if not landmarks_result or not landmarks_result.hand_landmarks:
        return frame

    h, w = frame.shape[:2]
    lm = landmarks_result.hand_landmarks[0]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    for start, end in connections:
        x1, y1 = int(lm[start].x * w), int(lm[start].y * h)
        x2, y2 = int(lm[end].x * w), int(lm[end].y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)

    fingertips = {4, 8, 12, 16, 20}

    for i, point in enumerate(lm):
        x, y = int(point.x * w), int(point.y * h)

        color = (59, 130, 246) if i in fingertips else (16, 185, 129)
        radius = 6 if i in fingertips else 4

        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)

    return frame


# ─────────────────────────────────────────────
# APPLY TOKEN TO SENTENCE
# ─────────────────────────────────────────────
def apply_token(sentence_chars, token):
    if token == "nothing":
        return

    if token == "space":
        if sentence_chars and sentence_chars[-1] != " ":
            sentence_chars.append(" ")
        return

    if token == "del":
        if sentence_chars:
            sentence_chars.pop()
        return

    sentence_chars.append(token.upper())


# ─────────────────────────────────────────────
# MAIN RECOGNIZER
# ─────────────────────────────────────────────
class ASLRecognizer:
    def __init__(
        self,
        model_path,
        labels_path,
        camera_id,
        min_conf,
        stable_frames,
        cooldown_sec
    ):

        # Load labels
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with open(labels_path, "r") as f:
            self.classes = json.load(f)

        print(f"Loaded {len(self.classes)} classes: {self.classes}")

        # Load CNN model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)

        print(f"Model loaded: {model_path}")
        print(f"Model input shape: {self.model.input_shape}")

        # MediaPipe
        self.landmarker = create_hand_landmarker()
        print("MediaPipe HandLandmarker ready")

        # Webcam
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Settings
        self.min_conf = min_conf
        self.stable_frames = stable_frames
        self.cooldown_sec = cooldown_sec

        # State
        self.pred_buffer = deque(maxlen=self.stable_frames)
        self.sentence_chars = []

        self.latest_token = "-"
        self.latest_conf = 0.0
        self.hand_detected = False
        self.top3 = []

        self.last_commit_time = 0.0
        self.last_committed_token = None

        # FPS
        self.fps_time = time.time()
        self.fps_counter = 0
        self.fps_value = 0.0

        self.lock = Lock()

    def clear_sentence(self):
        with self.lock:
            self.sentence_chars.clear()
            self.pred_buffer.clear()
            self.last_committed_token = None

    def get_status(self):
        with self.lock:
            return {
                "token": self.latest_token,
                "confidence": round(self.latest_conf * 100, 1),
                "sentence": "".join(self.sentence_chars),
                "fps": round(self.fps_value, 1),
                "hand_detected": self.hand_detected,
                "top3": self.top3,
            }

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass

        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except:
            pass

    # ─────────────────────────────────────────
    # FRAME PROCESSING
    # ─────────────────────────────────────────
    def process_frame(self, frame):
        """
        Process one webcam frame:
        1. Flip frame
        2. Detect hand using MediaPipe
        3. Crop hand region
        4. Send cropped image into CNN
        5. Draw result on screen
        """

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        token = "-"
        conf = 0.0
        top3 = []

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = self.landmarker.detect(mp_image)
        hand_detected = bool(result.hand_landmarks)

        if hand_detected:
            hand_input = crop_hand_from_result(frame, result)

            if hand_input is not None:
                probs = self.model.predict(hand_input, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                token = self.classes[pred_idx]

                top3_idx = probs.argsort()[-3:][::-1]
                top3 = [
                    {
                        "label": self.classes[i],
                        "conf": round(float(probs[i]) * 100, 1)
                    }
                    for i in top3_idx
                ]

            frame = draw_landmarks_on_frame(frame, result)

        # Update prediction state
        with self.lock:
            self.latest_token = token
            self.latest_conf = conf
            self.hand_detected = hand_detected
            self.top3 = top3

            if hand_detected and conf >= self.min_conf:
                self.pred_buffer.append(token)
            else:
                self.pred_buffer.clear()
                self.last_committed_token = None

            commit = None

            if len(self.pred_buffer) == self.stable_frames:
                winner, count = Counter(self.pred_buffer).most_common(1)[0]
                ratio = count / self.stable_frames
                time_ok = (time.time() - self.last_commit_time) >= self.cooldown_sec

                if ratio >= 0.7 and time_ok and winner != self.last_committed_token:
                    commit = winner

            if commit:
                apply_token(self.sentence_chars, commit)
                self.last_commit_time = time.time()
                self.last_committed_token = commit
                self.pred_buffer.clear()

            display_sentence = "".join(self.sentence_chars) or "-"

        # FPS counter
        self.fps_counter += 1
        elapsed = time.time() - self.fps_time

        if elapsed >= 1.0:
            self.fps_value = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_time = time.time()

        # Overlay UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        hand_color = (16, 185, 129) if hand_detected else (239, 68, 68)
        hand_text = "Hand detected" if hand_detected else "No hand"

        cv2.putText(
            frame,
            hand_text,
            (frame.shape[1] - 180, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            hand_color,
            2
        )

        cv2.putText(
            frame,
            f"Sign: {token}  ({conf * 100:.0f}%)",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Sentence: {display_sentence}",
            (16, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 220, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {self.fps_value:.1f}",
            (16, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 50),
            1
        )

        return frame

    def generate_mjpeg(self):
        while True:
            ok, frame = self.cap.read()

            if not ok:
                continue

            out = self.process_frame(frame)

            ok2, buf = cv2.imencode(
                ".jpg",
                out,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            if not ok2:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buf.tobytes() +
                b"\r\n"
            )


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
def create_app(recognizer):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
            recognizer.generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/status")
    def status():
        data = recognizer.get_status()
        sentence = data.get("sentence", "")
        data["suggestions"] = get_suggestions(sentence)
        return jsonify(data)

    @app.route("/delete_last", methods=["POST"])
    def delete_last():
        with recognizer.lock:
            if recognizer.sentence_chars:
                recognizer.sentence_chars.pop()

            recognizer.pred_buffer.clear()
            recognizer.last_committed_token = None

        return jsonify({"ok": True})

    @app.route("/clear", methods=["POST"])
    def clear():
        recognizer.clear_sentence()
        return jsonify({"ok": True})

    @app.route("/select_word", methods=["POST"])
    def select_word():
        req = request.get_json()
        selected_word = req.get("word", "")

        with recognizer.lock:
            sentence = "".join(recognizer.sentence_chars)
            words = sentence.split()

            if words:
                words[-1] = selected_word
            else:
                words.append(selected_word)

            # Add space after selected word automatically
            recognizer.sentence_chars = list(" ".join(words) + " ")

            recognizer.pred_buffer.clear()
            recognizer.last_committed_token = None

        return jsonify({"success": True})

    return app


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 50)
    print("  ASL Sign Language Recognition System")
    print("  FYP — Flask + MediaPipe + Real CNN")
    print("=" * 50)

    recognizer = ASLRecognizer(
        model_path=args.model_path,
        labels_path=args.labels_path,
        camera_id=args.camera_id,
        min_conf=args.min_conf,
        stable_frames=args.stable_frames,
        cooldown_sec=args.cooldown_sec,
    )

    app = create_app(recognizer)

    print("\nServer starting at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop\n")

    try:
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True
        )
    finally:
        recognizer.release()