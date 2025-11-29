"""
Lightweight MediaPipe Hands wrapper used by the letter demo.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """
    Simple helper around MediaPipe Hands that returns thumb and index tip
    landmarks in pixel coordinates.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._closed = False

    def process(self, frame_bgr: np.ndarray) -> Dict[str, object] | None:
        """
        Run hand tracking on a BGR frame.

        Returns:
            dict with thumb_tip, index_tip, and all landmarks in pixel coords, or
            None if no hand is detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks_px: List[Tuple[int, int]] = []
        for lm in hand_landmarks.landmark:
            x_px = int(lm.x * width)
            y_px = int(lm.y * height)
            landmarks_px.append((x_px, y_px))

        if len(landmarks_px) < 9:
            return None

        return {
            "thumb_tip": landmarks_px[4],
            "index_tip": landmarks_px[8],
            "landmarks": landmarks_px,
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        if not self._closed:
            self._hands.close()
            self._closed = True

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
