import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import uvicorn

# Load the trained classifier and scaler
with open('./pose/sign_pose_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('./pose/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Debugging: Check the type of the scaler
print(f"Scaler type: {type(scaler)}")

# Ensure scaler has transform method
if not hasattr(scaler, 'transform'):
    raise ValueError("Loaded scaler does not have a 'transform' method")

app = FastAPI()

def extract_landmarks_from_image(image: np.ndarray) -> np.ndarray:
    # Dummy implementation, replace with actual landmark extraction logic
    # For example, using a pre-trained model to detect landmarks
    landmarks = np.random.rand(21, 3).flatten()  # Example: 21 landmarks with x, y, z coordinates
    return landmarks

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    try:
        # Read image from the uploaded file
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Extract landmarks from the image
        landmarks = extract_landmarks_from_image(image)

        # Normalize the landmarks
        landmarks_scaled = scaler.transform([landmarks])

        # Make prediction
        prediction = classifier.predict(landmarks_scaled)

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
