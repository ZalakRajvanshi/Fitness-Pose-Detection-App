# src/app.py
import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from collections import deque
from tensorflow.keras.models import load_model
from utils import normalize_keypoints, angle_between

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/best_model.h5"
SEQ_LEN = 48
SQUAT_DOWN = 80
SQUAT_UP = 160
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Fitness Pose Tracker"
CSV_LOG_PATH = "results/session_log.csv"

# =========================
# Pose Utils
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class RepCounter:
    """Simple state machine for counting reps based on angle thresholds."""
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.state = 'up'
        self.count = 0

    def update(self, val):
        if self.state == 'up' and val <= self.low:
            self.state = 'down'
        elif self.state == 'down' and val >= self.high:
            self.state = 'up'
            self.count += 1
        return self.count


def extract_keypoints(results, w, h):
    """Extract pose landmarks into a flat (x, y) list."""
    if not results.pose_landmarks:
        return None
    return [(lm.x * w, lm.y * h) for lm in results.pose_landmarks.landmark]


def flatten_keypoints(kps):
    """Flatten keypoints into 1D numpy array."""
    return np.array(kps).flatten()


# =========================
# ML Inference
# =========================
def load_pose_model(model_path):
    try:
        model = load_model(model_path)
        print(f"[INFO] Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        return None


def run_inference(model, seq_buf):
    """Run ML classifier if buffer is full."""
    if model is None or len(seq_buf) < SEQ_LEN:
        return ""
    X = np.array(seq_buf)[None, :, :]
    prob = float(model.predict(X, verbose=0)[0, 0])
    label = "Correct" if prob >= 0.5 else "Incorrect"
    return f"Form (ML): {label} (p={prob:.2f})"


# =========================
# Session Logger
# =========================
def init_logger(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "squat_count", "knee_angle_L", "knee_angle_R",
                         "back_angle", "depth_ok", "form_ml"])
    print(f"[INFO] Logging session to {path}")


def log_entry(path, data):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)


# =========================
# Main App
# =========================
def main():
    model = load_pose_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    seq_buf = deque(maxlen=SEQ_LEN)
    squat_counter = RepCounter(SQUAT_DOWN, SQUAT_UP)
    init_logger(CSV_LOG_PATH)

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            kps = extract_keypoints(results, w, h)
            if kps:
                norm = normalize_keypoints(kps)
                seq_buf.append(norm)

                # Angles
                try:
                    hip_r, knee_r, ankle_r = kps[24], kps[26], kps[28]
                    hip_l, knee_l, ankle_l = kps[23], kps[25], kps[27]
                    shoulder, hip, knee = kps[11], kps[23], kps[25]

                    knee_r_angle = angle_between(hip_r, knee_r, ankle_r)
                    knee_l_angle = angle_between(hip_l, knee_l, ankle_l)
                    back_angle = angle_between(shoulder, hip, knee)
                except Exception:
                    knee_r_angle = knee_l_angle = back_angle = 180

                count = squat_counter.update((knee_r_angle + knee_l_angle) / 2)

                # Depth check (hip below knee ~ squat depth)
                depth_ok = knee_r_angle < 100 or knee_l_angle < 100

                # ML form check
                ml_msg = run_inference(model, seq_buf)

                # Text overlays
                cv2.putText(frame, f"Squat count: {count}", (10, 30), FONT, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Knee L/R: {int(knee_l_angle)} / {int(knee_r_angle)}",
                            (10, 60), FONT, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Back angle: {int(back_angle)}", (10, 90), FONT, 0.7, (200, 200, 255), 2)
                cv2.putText(frame, f"Depth OK: {'Yes' if depth_ok else 'No'}",
                            (10, 120), FONT, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, ml_msg, (10, 150), FONT, 0.6, (255, 128, 0), 2)

                # Log entry
                log_entry(CSV_LOG_PATH, [
                    time.time(), count, knee_l_angle, knee_r_angle,
                    back_angle, int(depth_ok), ml_msg
                ])

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
