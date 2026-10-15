import math
import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

EXTEND_MARGIN = 1.1  # fingertip must be this much farther from the wrist than its middle joint
THUMB_MARGIN = 1.05 # same concept but from base of pinky

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# finger name: (middle joint index, tip index)
FINGERS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}

def dist(a, b):
    """Straight-line distance between two (x, y) pixel points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(pts):
    """Returns {finger: True if extended, False if curled}."""
    wrist = pts[0]
    states = {}

    # thumb: tip farther from the pinky base than its middle joint is
    states["thumb"] = dist(pts[4], pts[17]) > dist(pts[3], pts[17]) * THUMB_MARGIN

    # other fingers: tip farther from the wrist than the middle joint is
    for name, (joint, tip) in FINGERS.items():
        states[name] = dist(pts[tip], wrist) > dist(pts[joint], wrist) * EXTEND_MARGIN

    return states

@dataclass # for the sole purpose of storing data, automatically initialised with __init__ and whatnot
class Hand:
    """Everything we know about one detected hand."""
    pts: list      # 21 pixel point values (x, y)
    states: dict   # finger details: {"thumb": True, etc}

class HandTracker:
    """Give it a webcam frame, get back a Hand (or None if no hand is visible)."""

    def __init__(self, model_path=MODEL_PATH, num_hands=1):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        h, w = frame.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in result.hand_landmarks[0]]
        return Hand(pts=pts, states=finger_states(pts))