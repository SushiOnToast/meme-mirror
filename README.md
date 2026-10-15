# Meme Mirror v1

A small Python project that watches your face and hands through your webcam and shows a matching meme in a second window, right next to the live camera feed.

Make an expression or a hand gesture, and the (cat) meme for it pops up.

## What it detects

**Face expressions**

| Expression | How to trigger it |
|---|---|
| Neutral | Relaxed face |
| Happy | Closed-mouth smile |
| Teeth | Big smile with teeth showing |
| Surprised | Mouth wide open with raised eyebrows |
| Eyebrow | Raise one eyebrow |
| Side | Turn your head to the side |

**Hand gestures** (these take priority over the face)

| Gesture | How to trigger it |
|---|---|
| Nerd | Index finger up, other fingers curled |
| Cool | Shaka: thumb and pinky out, other fingers curled |

## How it works

- **OpenCV** reads the webcam and draws the windows.
- **MediaPipe Face Landmarker** gives 52 face "blendshape" scores (smile, jaw open, brow raise, etc.) plus face landmarks. Simple rules turn those scores into an expression.
- **MediaPipe Hand Landmarker** gives 21 points per hand. The code checks which fingers are extended or curled to recognize a gesture.
- On startup the program **calibrates** to your resting face, so the thresholds are measured relative to you.

## Project structure

```
meme-mirror/
├── main.py              # webcam loop, smoothing, meme display
├── expressions.py       # face detection and expression rules
├── hand_tracker.py      # hand detection and finger states
├── gestures.py          # list of gestures and their rules
├── face_landmarker.task # MediaPipe face model
├── hand_landmarker.task # MediaPipe hand model
└── memes/               # one image per expression or gesture
```

## Setup

1. Use Python 3.12 (MediaPipe doesn't support the newest Python versions yet).
2. Create a virtual environment and install the libraries:

   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Add your memes to the `memes/` folder as `.jpg` files, named after what they show:

   ```
   neutral.jpg  happy.jpg  teeth.jpg  surprised.jpg
   eyebrow.jpg  side.jpg   nerd.jpg  cool.jpg
   ```

   Any missing meme is replaced by a grey placeholder card with the label on it.

## Run it

```bash
python main.py
```

Two windows open: your camera feed with the detected label, and the meme.

When the program starts, keep a relaxed face and look at the camera for about a second while it calibrates.

| Key | Action |
|---|---|
| `c` | Recalibrate (use this if your lighting or position changes) |
| `q` | Quit |

## Adding a new gesture

1. Write a rule in `gestures.py` that takes a hand and returns `True` or `False`.
2. Add it to the `GESTURES` list.
3. Drop a matching `.jpg` into `memes/`.
