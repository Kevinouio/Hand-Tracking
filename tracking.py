import cv2
import numpy as np
import mediapipe as mp
import time
import string

# ---------- Letter contour utilities ----------

def letter_to_points(letter: str, n_points: int = 120) -> np.ndarray:
    """
    Convert a letter into a list of n_points along its contour,
    normalized to roughly [-1, 1] x [-1, 1].
    """
    # Create a small canvas to draw the letter
    img_size = 256
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Put the letter roughly centered
    cv2.putText(
        img,
        letter,
        (40, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        5,
        255,
        thickness=10,
        lineType=cv2.LINE_AA,
    )

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        # Fallback: just a circle if something went wrong
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        return pts.astype(np.float32)

    # Take the largest contour
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)

    # Sample n_points evenly along the contour
    if len(cnt) < n_points:
        idxs = np.linspace(0, len(cnt) - 1, len(cnt), dtype=int)
    else:
        idxs = np.linspace(0, len(cnt) - 1, n_points, dtype=int)
    sampled = cnt[idxs].astype(np.float32)

    # Normalize to [-1, 1]
    x = sampled[:, 0]
    y = sampled[:, 1]
    x = (x - x.mean())
    y = (y - y.mean())
    scale = max(x.max() - x.min(), y.max() - y.min(), 1.0)
    x = 2.0 * x / scale
    y = 2.0 * y / scale

    return np.stack([x, y], axis=1).astype(np.float32)


def make_circle_points(n_points: int = 120) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)


def smooth_polyline(points: np.ndarray, smooth_level: float) -> np.ndarray:
    """
    Simple moving-average smoothing. smooth_level in [0, 1].
    Larger smooth_level => more smoothing.
    """
    if smooth_level <= 0.01:
        return points

    # Window size from 1 to 15 (odd)
    max_window = 15
    win = int(1 + smooth_level * (max_window - 1))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return points

    half = win // 2
    n = len(points)
    smoothed = np.zeros_like(points)
    for i in range(n):
        # circular padding
        idxs = [(i + j) % n for j in range(-half, half + 1)]
        smoothed[i] = points[idxs].mean(axis=0)
    return smoothed


# ---------- Gesture to letter mapping ----------

LETTERS = string.ascii_uppercase
ANGLE_PER_LETTER = 360.0 / len(LETTERS)


def angle_to_letter(angle_deg: float) -> str:
    idx = int(angle_deg // ANGLE_PER_LETTER)
    idx = max(0, min(idx, len(LETTERS) - 1))
    return LETTERS[idx]


# ---------- Main demo ----------

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    n_points = 120
    circle_pts = make_circle_points(n_points)
    cached_letters = {}

    current_letter = "A"
    letter_pts = letter_to_points(current_letter, n_points)
    last_change_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for nicer UX
        h, w, _ = frame.shape

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        thumb = None
        index = None
        angle_deg = None
        smooth_level = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # MediaPipe indices: 4 = thumb tip, 8 = index tip
            lm_thumb = hand_landmarks.landmark[4]
            lm_index = hand_landmarks.landmark[8]

            tx, ty = int(lm_thumb.x * w), int(lm_thumb.y * h)
            ix, iy = int(lm_index.x * w), int(lm_index.y * h)

            thumb = (tx, ty)
            index = (ix, iy)

            # Draw markers
            cv2.circle(frame, thumb, 8, (0, 255, 0), -1)
            cv2.circle(frame, index, 8, (0, 255, 0), -1)
            cv2.line(frame, thumb, index, (0, 255, 255), 2)

            # Compute angle
            dx = ix - tx
            dy = iy - ty
            angle_rad = np.arctan2(-dy, dx)  # flip y for mathematical coords
            angle_deg = (np.degrees(angle_rad) + 360.0) % 360.0

            # Compute distance -> smoothness
            dist = np.linalg.norm(np.array([dx, dy], dtype=np.float32))
            max_dist = w * 0.5
            smooth_level = float(np.clip(dist / max_dist, 0.0, 1.0))

            new_letter = angle_to_letter(angle_deg)
            if new_letter != current_letter:
                current_letter = new_letter
                # cache letter contours so we don't recompute every time
                if current_letter not in cached_letters:
                    cached_letters[current_letter] = letter_to_points(current_letter, n_points)
                letter_pts = cached_letters[current_letter]
                last_change_time = time.time()

        # Apply smoothing to letter points based on finger distance
        smoothed_letter_pts = smooth_polyline(letter_pts, smooth_level)

        # Compute morph progress t from circle -> letter (0.5s animation)
        elapsed = time.time() - last_change_time
        t = max(0.0, min(elapsed / 0.5, 1.0))

        morphed = (1.0 - t) * circle_pts + t * smoothed_letter_pts

        # Map normalized coords to a region in the frame (top-right corner)
        center_x = int(w * 0.75)
        center_y = int(h * 0.5)
        scale = int(min(w, h) * 0.2)

        pts_px = np.zeros((len(morphed), 2), dtype=int)
        pts_px[:, 0] = (morphed[:, 0] * scale + center_x).astype(int)
        pts_px[:, 1] = (morphed[:, 1] * scale + center_y).astype(int)

        # Draw the polyline made of dots
        for i in range(len(pts_px) - 1):
            cv2.line(
                frame,
                tuple(pts_px[i]),
                tuple(pts_px[i + 1]),
                (255, 255, 0),
                2,
            )
        # Optionally close the shape
        cv2.line(frame, tuple(pts_px[-1]), tuple(pts_px[0]), (255, 255, 0), 2)

        # Also draw individual dots
        for p in pts_px:
            cv2.circle(frame, tuple(p), 2, (255, 255, 255), -1)

        # Overlay debug text
        if angle_deg is not None:
            cv2.putText(
                frame,
                f"Angle: {angle_deg:6.1f} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            frame,
            f"Letter: {current_letter}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 200, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Smoothness: {smooth_level:4.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        cv2.imshow("Hand Letter Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
