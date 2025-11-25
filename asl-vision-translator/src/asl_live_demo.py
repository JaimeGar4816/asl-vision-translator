import cv2
import mediapipe as mp
import numpy as np
import joblib

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MODEL_FILE = "asl_model.joblib"
STABLE_FRAMES = 12  # how long a prediction must stay the same to be accepted

# Motion J/Z settings
MOTION_MIN_POINTS = 15       # minimum samples to count as motion
MOTION_MIN_PATH_LEN = 0.15   # how big the motion must be (normalized units)

# Simple vocabulary for word suggestions (UPPERCASE)
VOCAB = [
    "HELLO", "HELP", "HEY", "HI",
    "GOOD", "GREAT", "COOL", "OKAY",
    "PLEASE", "THANK", "THANKS", "WELCOME",
    "YES", "NO",
    "LOVE", "LIKE", "HATE",
    "NAME", "WHAT", "WHEN", "WHERE", "WHY", "HOW",
    "HOME", "HOUSE", "SCHOOL", "WORK",
    "APPLE", "FOOD", "WATER", "MILK",
    "YOU", "ME", "THEY", "THEM", "WE", "US",
    "STOP", "GO", "WAIT"
]


def extract_features(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)


def get_current_word(text_chars):
    """Return the current word (last sequence of letters) in UPPERCASE."""
    if not text_chars:
        return ""
    text = "".join(text_chars)
    text = text.rstrip()
    if not text:
        return ""
    parts = text.split(" ")
    last = parts[-1]
    # Keep only alphabetic chars
    word = "".join(ch for ch in last if ch.isalpha())
    return word.upper()


def get_suggestions(prefix, vocab, max_suggestions=3):
    """Return up to max_suggestions words from vocab that start with prefix."""
    if not prefix or len(prefix) < 2:
        return []
    prefix = prefix.upper()
    matches = [w for w in vocab if w.startswith(prefix)]
    # Sort by length then alphabetically for nicer ordering
    matches.sort(key=lambda w: (len(w), w))
    return matches[:max_suggestions]


def apply_suggestion(text_chars, suggestion):
    """Replace the last word in text_chars with suggestion (and add a space)."""
    text = "".join(text_chars)
    text = text.rstrip()
    # Find last space
    last_space = text.rfind(" ")
    if last_space == -1:
        base = ""
    else:
        base = text[: last_space + 1]  # include the space

    new_text = base + suggestion  # don't auto-add space; let user decide with space key
    text_chars.clear()
    text_chars.extend(list(new_text))


def draw_side_panel(frame, text_chars, current_letter, motion_mode, motion_target, current_word, suggestions):
    h, w, _ = frame.shape
    panel_width = 280
    x0 = w - panel_width
    y0 = 0
    x1 = w
    y1 = h

    # Background panel
    cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 20, 20), -1)

    # Title
    cv2.putText(
        frame,
        "ASL ML Demo",
        (x0 + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Current letter (static ML)
    cv2.putText(
        frame,
        "Current:",
        (x0 + 20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        current_letter if current_letter else "-",
        (x0 + 20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0) if current_letter else (80, 80, 80),
        2,
        cv2.LINE_AA,
    )

    # Text output
    cv2.putText(
        frame,
        "Text:",
        (x0 + 20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    text = "".join(text_chars)
    max_chars_per_line = 18
    lines = [text[i: i + max_chars_per_line] for i in range(0, len(text), max_chars_per_line)]

    y_text = 155
    for line in lines[:8]:
        cv2.putText(
            frame,
            line,
            (x0 + 20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_text += 22

    # Current word & suggestions
    y_word = y_text + 5
    cv2.putText(
        frame,
        f"Word: {current_word if current_word else '-'}",
        (x0 + 20, y_word),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    y_sugg = y_word + 25
    cv2.putText(
        frame,
        "Suggestions:",
        (x0 + 20, y_sugg),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    y_sugg += 22

    for i, word in enumerate(suggestions[:3]):
        line = f"{i+1}: {word}"
        cv2.putText(
            frame,
            line,
            (x0 + 25, y_sugg),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_sugg += 20

    if not suggestions:
        cv2.putText(
            frame,
            "(type 2+ letters)",
            (x0 + 25, y_sugg),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        y_sugg += 18

    # Motion status
    if motion_mode:
        if motion_target:
            motion_text = f"{motion_target} (drawing)"
        else:
            motion_text = "Select J or Z..."
    else:
        motion_text = "Off"

    cv2.putText(
        frame,
        f"Motion J/Z: {motion_text}",
        (x0 + 20, y_sugg + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (0, 255, 255) if motion_mode else (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Instructions
    base_y = h - 105
    cv2.putText(
        frame,
        "space: add space",
        (x0 + 20, base_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "b: backspace   c: clear",
        (x0 + 20, base_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "q: quit   0: motion mode",
        (x0 + 20, base_y + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "J/Z, M to finish | 1-3: pick word",
        (x0 + 20, base_y + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def compute_path_length(points):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        dx = x2 - x1
        dy = y2 - y1
        total += (dx * dx + dy * dy) ** 0.5
    return total


def main():
    # Load trained model (A–I, K–Y)
    bundle = joblib.load(MODEL_FILE)
    clf = bundle["model"]
    le = bundle["label_encoder"]

    print("Loaded model. Classes:", list(le.classes_))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    last_pred = ""
    stable_count = 0
    current_letter = ""
    text_chars = []

    # Tracks the last letter actually added to text
    committed_letter = None

    # Motion mode state
    motion_mode = False
    motion_target = None  # "J" or "Z"
    motion_points = []    # list of (x, y) normalized

    # Last computed suggestions for keys 1–3
    last_suggestions = []

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
            predicted_letter = ""
            fingertip_xy = None

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 150, 0), thickness=2),
                )

                # Index fingertip for motion tracking
                tip = hand_landmarks.landmark[8]
                fingertip_xy = (tip.x, tip.y)

                if not motion_mode:
                    # Static prediction (A–I, K–Y)
                    feat = extract_features(hand_landmarks).reshape(1, -1)
                    pred_idx = clf.predict(feat)[0]
                    predicted_letter = le.inverse_transform([pred_idx])[0]

                    cv2.putText(
                        frame,
                        f"Pred: {predicted_letter}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    msg = "Motion: choose J or Z" if motion_target is None else f"Drawing {motion_target}..."
                    cv2.putText(
                        frame,
                        msg,
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

            # Track motion path for J/Z
            if motion_mode and fingertip_xy is not None and motion_target is not None:
                motion_points.append(fingertip_xy)
                for i in range(1, len(motion_points)):
                    x1, y1 = motion_points[i - 1]
                    x2, y2 = motion_points[i]
                    p1 = (int(x1 * w), int(y1 * h))
                    p2 = (int(x2 * w), int(y2 * h))
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Static stability logic (only when not in motion mode)
            if not motion_mode:
                # If prediction changed, reset stability and allow repeating letters again
                if predicted_letter == last_pred and predicted_letter != "":
                    stable_count += 1
                else:
                    stable_count = 0
                    if predicted_letter != last_pred:
                        committed_letter = None
                    last_pred = predicted_letter

                # When stable long enough, commit letter
                if stable_count == STABLE_FRAMES and predicted_letter != "":
                    if committed_letter != predicted_letter:
                        text_chars.append(predicted_letter)
                        committed_letter = predicted_letter
                        print(f"Committed letter: {predicted_letter}")
                    stable_count = 0

                current_letter = predicted_letter
            else:
                current_letter = ""

            # Compute current word + suggestions
            current_word = get_current_word(text_chars)
            suggestions = get_suggestions(current_word, VOCAB)
            last_suggestions = suggestions  # store for 1–3 keys

            # Draw side panel
            draw_side_panel(frame, text_chars, current_letter, motion_mode, motion_target, current_word, suggestions)

            cv2.imshow("ASL Live ML Demo + Motion J/Z + Suggestions", frame)
            key = cv2.waitKey(1) & 0xFF

            # Keyboard controls
            if key == ord("q"):
                break
            elif key == ord("c"):
                text_chars = []
            elif key == ord("b"):
                if text_chars:
                    text_chars.pop()
            elif key == ord(" "):
                text_chars.append(" ")

            # Accept suggestion 1/2/3
            elif key in (ord("1"), ord("2"), ord("3")):
                idx = key - ord("1")
                if 0 <= idx < len(last_suggestions):
                    chosen = last_suggestions[idx]
                    print(f"Applied suggestion: {chosen}")
                    apply_suggestion(text_chars, chosen)

            # Start motion mode (0)
            elif key == ord("0"):
                motion_mode = True
                motion_target = None
                motion_points = []
                print("Motion mode activated. Press J for J-motion or Z for Z-motion.")

            # Choose J or Z while in motion mode
            elif motion_mode and key in (ord("j"), ord("J")):
                motion_target = "J"
                motion_points = []
                print("Drawing J... move your fingertip, then press M to finish.")
            elif motion_mode and key in (ord("z"), ord("Z")):
                motion_target = "Z"
                motion_points = []
                print("Drawing Z... move your fingertip, then press M to finish.")

            # Finish motion (M)
            elif key in (ord("m"), ord("M")):
                if motion_mode and motion_target and len(motion_points) >= MOTION_MIN_POINTS:
                    path_len = compute_path_length(motion_points)
                    print(f"Motion for {motion_target} path length:", path_len)

                    if path_len >= MOTION_MIN_PATH_LEN:
                        text_chars.append(motion_target)
                        print(f"Added motion-based letter: {motion_target}")
                    else:
                        print("Motion too small, not adding letter.")

                # Reset motion state
                motion_mode = False
                motion_target = None
                motion_points = []

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
