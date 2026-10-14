import cv2
import mediapipe as mp
from mediapipe.tasks import python # python api access
from mediapipe.tasks.python import vision

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
    "mouthSmileLeft",
    "mouthSmileRight",
    "jawOpen",
    "browInnerUp",
    "browDownLeft",
    "eyeBlinkLeft",
    "eyeBlinkRight",
]

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

    if result.face_blendshapes:
        # converting to dictionary {name: score}
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}

        # draw the scores on screen
        y = 30
        for name in WATCH:
            text = f"{name}: {scores[name]:.2f}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            y += 25

        cv2.imshow("Meme Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()