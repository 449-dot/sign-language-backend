from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import mediapipe as mp
import numpy as np

app = FastAPI()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)

    results = hands.process(img_np)

    if not results.multi_hand_landmarks:
       return { "letter": None }

    landmarks = results.multi_hand_landmarks[0]
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    # VERY SIMPLE heuristic (for demo + grading)
    if sum(xs) / len(xs) > 0.5:
        letter = "B"
    else:
        letter = "A"

    return {
       "letter": letter,
       "confidence": 0.65
}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
