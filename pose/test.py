import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the trained classifier and scaler
with open('sign_pose_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Take live camera input for pose detection
video_path = "C:/Users/KWAME/Downloads//Greetings in ASL _ ASL - American Sign Language.mp4"
cap = cv2.VideoCapture(video_path)

def preprocess_landmarks(landmarks):
    data = []
    for landmark in landmarks.landmark:
        data.extend([landmark.x, landmark.y, landmark.z])
    return data

# Main loop for real-time prediction
while True:
    ret, img = cap.read()
    img = cv2.resize(img, (600, 400))
    results = pose.process(img)

    if results.pose_landmarks:
        landmarks_data = preprocess_landmarks(results.pose_landmarks)
        landmarks_data_scaled = scaler.transform([landmarks_data])
        predicted_sign = classifier.predict(landmarks_data_scaled)
        print(f"Predicted sign: {predicted_sign[0]}")

        # Display the video feed with pose landmarks and prediction
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(img, f"Predicted sign: {predicted_sign[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Pose Estimation", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
