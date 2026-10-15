import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# every label the classifier can return (main.py loads one meme per label)
LABELS = ["eyebrow", "happy", "teeth", "surprised", "side", "neutral"]

# thresholds for detection
SMILE = 0.45 # average smile score for happy
TEETH_JAW = 0.12 # slightly parted lips means teeth showing
TEETH_UPPER_LIP = 0.20 # upper lip lifting means teeth showing
SURPRISE_JAW = 0.35 # must be sufficiently open to pass as surprised
SURPRISE_BROW = 0.40
EYEBROW_RAISE = 0.06 # minimum increase for the raised eyebrow
EYEBROW_LOW = 0.04 # maximum increase for the eyebrow that stays down
YAW_LIMIT = 0.3 # how far the nose moves off-center for side profile (left, right)

CALIBRATION_FRAMES = 30
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")


def head_turn(lm):
    """0 when facing the camera, positive/negative as the head turns."""
    left, right, nose = lm[234].x, lm[454].x, lm[1].x # edge of face landmarks and nose
    width = right - left
    if abs(width) < 1e-6:
        return 0.0
    return (nose - left) / width - 0.5 # where the nose stands between the two, 0.5 = centre


def detect_raised_eyebrow(scores, baseline):
    """Detect one raised eyebrow while the other stays relatively low."""
    left_up = scores["browOuterUpLeft"] - baseline.get("browOuterUpLeft", 0.0)
    right_up = scores["browOuterUpRight"] - baseline.get("browOuterUpRight", 0.0)
    left_down = scores["browDownLeft"] - baseline.get("browDownLeft", 0.0)
    right_down = scores["browDownRight"] - baseline.get("browDownRight", 0.0)

    left_raised = (
        left_up > EYEBROW_RAISE
        and right_up < EYEBROW_LOW
        and right_down >= left_down
    )
    right_raised = (
        right_up > EYEBROW_RAISE
        and left_up < EYEBROW_LOW
        and left_down >= right_down
    )
    return left_raised or right_raised


def classify(s, yaw, calibrated, raw=None, baseline=None):
    smile = (s["mouthSmileLeft"] + s["mouthSmileRight"]) / 2
    upper_lip = (s["mouthUpperUpLeft"] + s["mouthUpperUpRight"]) / 2
    jaw = s["jawOpen"]
    brow_up = s["browInnerUp"]

    if abs(yaw) > YAW_LIMIT:
        return "side"

    if calibrated and raw is not None and baseline is not None:
        if detect_raised_eyebrow(raw, baseline):
            return "eyebrow"

    if jaw > SURPRISE_JAW and brow_up > SURPRISE_BROW:
        return "surprised"

    if smile > SMILE and (jaw > TEETH_JAW or upper_lip > TEETH_UPPER_LIP):
        return "teeth"

    if smile > SMILE:
        return "happy"

    return "neutral"


class ExpressionDetector:
    """Give it a webcam frame, get back an expression label."""

    def __init__(self, model_path=MODEL_PATH):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

        self.baseline = {}
        self.calibrating = True  # calibrate automatically on startup
        self._calib_frames = []

    def recalibrate(self):
        self.calibrating = True
        self._calib_frames = []

    def process(self, frame):
        """Takes a BGR frame, returns (label, info_dict)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not (result.face_blendshapes and result.face_landmarks):
            return ("calibrating... relax" if self.calibrating else "no face"), {}

        lm = result.face_landmarks[0]
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}

        if self.calibrating:
            self._calib_frames.append(scores)
            if len(self._calib_frames) >= CALIBRATION_FRAMES:
                n = len(self._calib_frames)
                self.baseline = {
                    k: sum(f[k] for f in self._calib_frames) / n for k in scores
                }
                self.calibrating = False
                self._calib_frames = []

        adjusted = {
            k: max(0.0, v - self.baseline.get(k, 0.0)) for k, v in scores.items()
        }
        yaw = head_turn(lm)

        info = {
            "smile": (adjusted["mouthSmileLeft"] + adjusted["mouthSmileRight"]) / 2,
            "jaw": adjusted["jawOpen"],
            "browUp": adjusted["browInnerUp"],
            "leftBrow": scores["browOuterUpLeft"] - self.baseline.get("browOuterUpLeft", 0.0),
            "rightBrow": scores["browOuterUpRight"] - self.baseline.get("browOuterUpRight", 0.0),
            "yaw": yaw,
        }

        if self.calibrating:
            return "calibrating... relax", info

        label = classify(adjusted, yaw, calibrated=bool(self.baseline),
                         raw=scores, baseline=self.baseline)
        return label, info