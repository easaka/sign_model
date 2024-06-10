from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
from io import BytesIO


app = FastAPI()

# Load the trained model and labels dictionary
model_dict = pickle.load(open('./sign/model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

class PredictionResponse(BaseModel):
    predicted_character: str

def process_image(image: Image.Image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        raise HTTPException(status_code=400, detail="No hand detected in the image.")

    data_aux = []
    x_ = []
    y_ = []

    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

    prediction = model.predict([np.asarray(data_aux)])
    predicted_character = labels_dict[int(prediction[0])]
    return predicted_character

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file as an image
        image = Image.open(io.BytesIO(await file.read()))
        predicted_character = process_image(image)
        return {"predicted_character": predicted_character}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Load the trained classifier and scaler
with open('./pose/sign_pose_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('./pose/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class PredictionResponse(BaseModel):
    predicted_sign: str

def preprocess_landmarks(landmarks):
    data = []
    for landmark in landmarks.landmark:
        data.extend([landmark.x, landmark.y, landmark.z])
    return data

@app.post("/predict/pose", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded video file
        contents = await file.read()
        video_path = 'uploaded_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(contents)

        cap = cv2.VideoCapture(video_path)

        while True:
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.resize(img, (600, 400))
            results = pose.process(img)

            if results.pose_landmarks:
                landmarks_data = preprocess_landmarks(results.pose_landmarks)
                landmarks_data_scaled = scaler.transform([landmarks_data])
                predicted_sign = classifier.predict(landmarks_data_scaled)
                predicted_sign = predicted_sign[0]
                break

        cap.release()
        return {"predicted_sign": predicted_sign}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
