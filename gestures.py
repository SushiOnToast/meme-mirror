def nerd(hand):
    """Index finger up, everything else curled."""
    s = hand.states
    return (
        s["index"]
        and not s["thumb"]
        and not s["middle"]
        and not s["ring"]
        and not s["pinky"]
    )

def cool(hand):
    """Thumb and pinky out, the other three fingers curled (the shaka)."""
    s = hand.states
    return (
        s["thumb"]
        and s["pinky"]
        and not s["index"]
        and not s["middle"]
        and not s["ring"]
    )

# Every gesture lives in this list as (label, rule), rule is a function
GESTURES = [
    ("nerd", nerd),
    ("cool", cool)
]

GESTURE_LABELS = [name for name, _ in GESTURES]


def classify_gesture(hand):
    """Takes a Hand (or None), returns a gesture label or "none"."""
    if hand is None:
        return "none"
    for name, rule in GESTURES:
        if rule(hand):
            return name
    return "none"