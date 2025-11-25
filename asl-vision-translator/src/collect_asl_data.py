import cv2
import mediapipe as mp
import numpy as np
import os
from collections import Counter

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Supported letters: A–Z except J and Z (motion-based in live demo)
VALID_LABELS = [ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if ch not in ("J", "Z")]

# How many samples we want per letter (you can tweak this)
TARGET_PER_LETTER = 30


def extract_features(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)


def build_counts_dict(y_labels):
    counts = Counter(y_labels)
    return {label: counts.get(label, 0) for label in VALID_LABELS}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    X = []
    y = []
    current_label = ""

    # Load existing dataset if it exists (append mode)
    if os.path.exists("asl_data.npz"):
        data = np.load("asl_data.npz")
        X_old = data["X"]
        y_old = data["y"]
        X = list(X_old)
        y = list(y_old)
        print("Loaded existing dataset: asl_data.npz")
        print("  Existing total samples:", len(y))
    else:
        print("No existing dataset found. Starting fresh.")

    # Build initial per-letter counts
    counts = build_counts_dict(y)

    print("\nASL Data Collector (SAVE = 6, QUIT = 1)")
    print("----------------------------------------")
    print("Supported static letters (no J/Z):", " ".join(VALID_LABELS))
    print("Controls:")
    print("  Press A–Z (except J/Z) to choose label.")
    print("  Press 6 to SAVE the current hand sample.")
    print("  Press 1 to quit and save asl_data.npz.\n")

    print("Current counts per letter:")
    for lbl in VALID_LABELS:
        print(f"  {lbl}: {counts[lbl]}")
    need_list = [lbl for lbl in VALID_LABELS if counts[lbl] < TARGET_PER_LETTER]
    print(f"\nLetters still below {TARGET_PER_LETTER} samples:", " ".join(need_list) if need_list else "None 🎉")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            h, w, _ = frame.shape
            hand_landmarks = None

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 150, 0), thickness=2),
                )

                cv2.putText(
                    frame,
                    "Hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "Show one hand to the camera",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Current label & total samples
            cv2.putText(
                frame,
                f"Label: {current_label if current_label else '-'}",
                (10, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Total samples: {len(y)}",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # --- Per-letter counts UI block ---
            start_x = int(w * 0.45)
            start_y = 40
            row_height = 20
            col_width = 80
            letters_per_row = 6

            cv2.putText(
                frame,
                f"Counts (target {TARGET_PER_LETTER}):",
                (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            for idx, lbl in enumerate(VALID_LABELS):
                row = idx // letters_per_row
                col = idx % letters_per_row
                x = start_x + col * col_width
                y_txt = start_y + row * row_height
                txt = f"{lbl}:{counts[lbl]:02d}"
                color = (0, 255, 0) if counts[lbl] >= TARGET_PER_LETTER else (180, 180, 180)
                cv2.putText(
                    frame,
                    txt,
                    (x, y_txt),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # Line with letters that still need samples
            need_letters = [lbl for lbl in VALID_LABELS if counts[lbl] < TARGET_PER_LETTER]
            need_text = "Need: " + (" ".join(need_letters) if need_letters else "All reached target!")
            cv2.putText(
                frame,
                need_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if not need_letters else (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ASL Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            # QUIT (key "1")
            if key == ord("1"):
                break

            # Choose letter A–Z except J/Z
            if (ord("a") <= key <= ord("z")) or (ord("A") <= key <= ord("Z")):
                ch = chr(key).upper()
                if ch in VALID_LABELS:
                    current_label = ch
                    print(f"\nLabel set to: {current_label}")
                    print(f"Current count for {current_label}: {counts[current_label]}")
                else:
                    print(f"'{ch}' ignored (J/Z are motion-based in the live demo).")

            # SAVE SAMPLE using key "6"
            if key == ord("6"):
                if current_label == "":
                    print("Pick a label first (A–Z except J/Z).")
                elif hand_landmarks is None:
                    print("No hand detected, can't save.")
                else:
                    feat = extract_features(hand_landmarks)
                    X.append(feat)
                    y.append(current_label)
                    counts[current_label] += 1

                    print(f"Saved sample for {current_label}. Total for {current_label} = {counts[current_label]}")
                    need_list = [lbl for lbl in VALID_LABELS if counts[lbl] < TARGET_PER_LETTER]
                    print(f"Total samples: {len(y)}")
                    print(f"Letters still below {TARGET_PER_LETTER} samples:", " ".join(need_list) if need_list else "None 🎉")

        cap.release()
        cv2.destroyAllWindows()

    if len(y) == 0:
        print("No samples collected.")
        return

    X_arr = np.array(X)
    y_arr = np.array(y)
    np.savez("asl_data.npz", X=X_arr, y=y_arr)
    print("\nSaved combined dataset to asl_data.npz")
    print("  Final total samples:", len(y_arr))
    final_counts = build_counts_dict(y_arr)
    print("  Final per-letter counts:")
    for lbl in VALID_LABELS:
        print(f"    {lbl}: {final_counts[lbl]}")


if __name__ == "__main__":
    main()
