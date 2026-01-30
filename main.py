from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import pickle
import mediapipe as mp
import numpy as np

app = FastAPI()


# --------------------------------------------------
# CORS (REQUIRED for browser + WebView)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # localhost, Netlify, Android WebView
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load trained model
# -----------------------------
with open("model.pkl", "rb") as f:
    model_dict = pickle.load(f)
    model = model_dict["model"]

# -----------------------------
# Labels dictionary (FULL)
# -----------------------------
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3',
    30: '4', 31: '5', 32: '6', 33: '7',
    34: '8', 35: '9', 36: ' ', 37: '.'
}

EXPECTED_FEATURES = 42

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Sign Language API is running",
        "endpoint": "/predict"
    }

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)

    results = hands.process(img_np)

    if not results.multi_hand_landmarks:
        return {
            "letter": None,
            "confidence": 0.0
        }

    hand_landmarks = results.multi_hand_landmarks[0]

    x_ = []
    y_ = []
    data_aux = []

    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)

    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min(x_))
        data_aux.append(lm.y - min(y_))

    # Pad or trim to expected length
    if len(data_aux) < EXPECTED_FEATURES:
        data_aux.extend([0] * (EXPECTED_FEATURES - len(data_aux)))
    else:
        data_aux = data_aux[:EXPECTED_FEATURES]

    # Model prediction
    prediction = model.predict([np.asarray(data_aux)])
    pred_class = int(prediction[0])

    predicted_letter = labels_dict.get(pred_class)

    if predicted_letter is None:
        return {
            "letter": None,
            "confidence": 0.0
        }

    return {
        "letter": predicted_letter,
        "confidence": 0.65
    }
