import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
DATA_DIR = "data"

def collect_data(label, samples=50, seq_len=48):
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        data = []
        while len(data) < samples:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                kps = []
                for lm in results.pose_landmarks.landmark:
                    kps.extend([lm.x, lm.y])  # 2D keypoints
                data.append(kps)
                cv2.putText(frame, f"Collected: {len(data)}/{samples}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Collecting", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

    npy_path = os.path.join(DATA_DIR, f"{label}_{len(data)}.npy")
    np.save(npy_path, np.array(data))
    print(f"Saved {npy_path}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    # Example: collect correct pushups (label=1), incorrect pushups (label=0)
    collect_data(label="pushup_correct", samples=100)
    collect_data(label="pushup_incorrect", samples=100)
