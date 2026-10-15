import cv2
from hand_tracker import HandTracker
from gestures import classify_gesture

tracker = HandTracker()
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    hand = tracker.process(frame)
    gesture = classify_gesture(hand)

    if hand:
        for x, y in hand.pts:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        y = 30
        for name, up in hand.states.items():
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