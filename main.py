import os
import cv2
import numpy as np
from expressions import ExpressionDetector, LABELS

DEBUG = True  

MEME_SIZE = 480
MEME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memes")


def make_placeholder(text, size=MEME_SIZE):
    """A plain grey card with text, used when a meme file is missing."""
    img = np.full((size, size, 3), 60, dtype=np.uint8)
    cv2.putText(img, text.upper(), (20, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def fit_to_canvas(img, size=MEME_SIZE):
    """Shrink or grow an image to fit a square canvas without stretching it."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def load_memes():
    """Load every meme once and return a {label: image} dictionary."""
    memes = {}
    for label in LABELS:
        filename = label.replace(" ", "_") + ".jpg"
        img = cv2.imread(os.path.join(MEME_DIR, filename))
        if img is None:
            print(f"Missing {filename}, using a placeholder")
            memes[label] = make_placeholder(label)
        else:
            memes[label] = fit_to_canvas(img)
    return memes


detector = ExpressionDetector()
memes = load_memes()
blank = make_placeholder("...")

cv2.namedWindow("Meme")
cv2.moveWindow("Meme", 700, 50)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    label, info = detector.process(frame)

    if DEBUG:
        y = 30
        for name, value in info.items():
            cv2.putText(frame, f"{name} {value:+.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25

    cv2.putText(frame, label.upper(), (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Meme Mirror", frame)
    cv2.imshow("Meme", memes.get(label, blank))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        detector.recalibrate()

cap.release()
cv2.destroyAllWindows()