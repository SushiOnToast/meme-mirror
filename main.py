import cv2
import mediapipe as mp
from mediapipe.tasks import python # python api access
from mediapipe.tasks.python import vision

# thresholds for detection
SMILE = 0.45            # average smile score for happy
TEETH_JAW = 0.12        # slightly parted lips => teeth showing
TEETH_UPPER_LIP = 0.20  # upper lip lifting => teeth showing
SURPRISE_JAW = 0.35
SURPRISE_BROW = 0.40
EYEBROW_RAISE = 0.06    # minimum increase for the raised eyebrow
EYEBROW_LOW = 0.04      # maximum increase for the eyebrow that stays down
YAW_LIMIT = 0.18        # how far the nose moves off-center for side profile
DEBUG = True            # show numbers

# load the face model and ask it to output blendshape scores
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# choose blendshapes
WATCH = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "mouthFrownLeft",
    "mouthFrownRight",
    "eyeSquintLeft",
    "eyeSquintRight",
]


def head_turn(lm):
    """0 when facing the camera, positive/negative as the head turns."""
    left, right, nose = lm[234].x, lm[454].x, lm[1].x
    width = right - left
    if abs(width) < 1e-6:
        return 0.0
    return (nose - left) / width - 0.5


def detect_raised_eyebrow(scores, baseline):
    """Detect one raised eyebrow while the other stays relatively low."""
    left_up = (
        scores["browOuterUpLeft"]
        - baseline.get("browOuterUpLeft", 0.0)
    )
    right_up = (
        scores["browOuterUpRight"]
        - baseline.get("browOuterUpRight", 0.0)
    )

    left_down = (
        scores["browDownLeft"]
        - baseline.get("browDownLeft", 0.0)
    )
    right_down = (
        scores["browDownRight"]
        - baseline.get("browDownRight", 0.0)
    )

    # One eyebrow must rise clearly more than the other.
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
        return "side profile"

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


baseline = {}
calib_frames = []
calibrating = True  # calibrate automatically on startup

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    # converting bgr to rgb format
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) # container class to process video frame
    result = detector.detect(mp_image)

    label = "no face"

    if result.face_blendshapes and result.face_landmarks:
        lm = result.face_landmarks[0]
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}

        if calibrating:
            calib_frames.append(scores)
            if len(calib_frames) >= 30:
                baseline = {
                    k: sum(f[k] for f in calib_frames) / len(calib_frames)
                    for k in scores
                }
                calibrating = False
                calib_frames = []

        adjusted = {
            k: max(0.0, v - baseline.get(k, 0.0))
            for k, v in scores.items()
        }

        yaw = head_turn(lm)
        label = classify(
            adjusted,
            yaw,
            calibrated=bool(baseline),
            raw=scores,
            baseline=baseline
        )

        if DEBUG:
            smile = (adjusted["mouthSmileLeft"] + adjusted["mouthSmileRight"]) / 2
            left_brow = (
                scores["browOuterUpLeft"]
                - baseline.get("browOuterUpLeft", 0.0)
            )
            right_brow = (
                scores["browOuterUpRight"]
                - baseline.get("browOuterUpRight", 0.0)
            )

            info = [
                f"smile {smile:.2f}",
                f"jaw {adjusted['jawOpen']:.2f}",
                f"browUp {adjusted['browInnerUp']:.2f}",
                f"leftBrow {left_brow:+.2f}",
                f"rightBrow {right_brow:+.2f}",
                f"yaw {yaw:+.2f}",
            ]
            y = 30
            for line in info:
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                y += 25

    if calibrating:
        label = "calibrating... relax"

    cv2.putText(frame, label.upper(), (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Meme Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()