from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import mediapipe as mp
import numpy as np

# -------------------
# App initialization
# -------------------
app = FastAPI()

# -------------------
# CORS (REQUIRED for browser + WebView)
# -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for school project
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# MediaPipe Hands
# -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -------------------
# Health check route
# -------------------
@app.get("/")
def root():
    return {
        "message": "Sign Language API is running",
        "endpoint": "/predict"
    }

# -------------------
# Prediction endpoint
# -------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # MediaPipe expects RGB numpy array
    img_np = np.array(image)

    results = hands.process(img_np)

    if not results.multi_hand_landmarks:
        return {
            "letter": None,
            "confidence": 0.0
        }

    landmarks = results.multi_hand_landmarks[0]
    xs = [lm.x for lm in landmarks.landmark]

    # VERY SIMPLE heuristic (demo)
    letter = "B" if sum(xs) / len(xs) > 0.5 else "A"

    return {
        "letter": letter,
        "confidence": 0.65
    }
