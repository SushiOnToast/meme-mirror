import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

EXTEND_MARGIN = 1.1  # fingertip must be this much farther from the wrist than its middle joint
THUMB_MARGIN = 1.05 # same concept but from base of pinky

# finger name: (middle joint index, tip index)
FINGERS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)


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


def is_well_actually(states):
    return (
        states["index"]
        and not states["thumb"]
        and not states["middle"]
        and not states["ring"]
        and not states["pinky"]
    )


cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    gesture = "none"

    if result.hand_landmarks:
        h, w = frame.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in result.hand_landmarks[0]]

        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        states = finger_states(pts)
        if is_well_actually(states):
            gesture = "well actually"

        y = 30
        for name, up in states.items():
            cv2.putText(frame, f"{name}: {'UP' if up else 'curled'}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25

    cv2.putText(frame, gesture.upper(), (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Hand Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()