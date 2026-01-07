from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import mediapipe as mp
import numpy as np
import pickle

app = FastAPI()

# ---------------- LOAD MODEL ----------------
with open("model.p", "rb") as f:
    model_dict = pickle.load(f)
    model = model_dict["model"]

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
    23: 'X', 24: 'Y', 25: 'Z'
}

EXPECTED_FEATURES = 42

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {
        "message": "Sign Language API is running",
        "endpoint": "/predict"
    }

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

    # ---- FEATURE EXTRACTION (MATCH TRAINING) ----
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]

    data_aux = []
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min(x_))
        data_aux.append(lm.y - min(y_))

    # Padding safety
    if len(data_aux) < EXPECTED_FEATURES:
        data_aux.extend([0] * (EXPECTED_FEATURES - len(data_aux)))
    elif len(data_aux) > EXPECTED_FEATURES:
        data_aux = data_aux[:EXPECTED_FEATURES]

    # ---- PREDICTION ----
    prediction = model.predict([np.asarray(data_aux)])
    predicted_letter = labels_dict[int(prediction[0])]

    confidence = (
        max(model.predict_proba([data_aux])[0])
        if hasattr(model, "predict_proba")
        else 1.0
    )

    return {
        "letter": predicted_letter,
        "confidence": round(float(confidence), 2)
    }
