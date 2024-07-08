from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import uvicorn
import cv2
import mediapipe as mp
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained classifier and scaler for pose signs
with open('../pose/sign_pose_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('../pose/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class PosePredictionResponse(BaseModel):
    predicted_signs: list

def preprocess_landmarks(landmarks):
    data = []
    for landmark in landmarks.landmark:
        data.extend([landmark.x, landmark.y, landmark.z])
    return data

@app.websocket("/ws/predict/pose")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            np_data = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            img = cv2.resize(img, (600, 400))
            results = pose.process(img)

            if results.pose_landmarks:
                landmarks_data = preprocess_landmarks(results.pose_landmarks)
                landmarks_data_scaled = scaler.transform([landmarks_data])
                predicted_sign = classifier.predict(landmarks_data_scaled)
                await websocket.send_json({"predicted_signs": [predicted_sign[0]]})
            else:
                await websocket.send_json({"predicted_signs": []})
    except Exception as e:
        print(f"Error in WebSocket: {e}")
    finally:
        await websocket.close()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=8000)
