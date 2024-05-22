import cv2
import mediapipe as mp
import numpy as np
import os
import json

# Initialize mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create a directory to save landmark data
data_dir = "sign_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define a list to store the collected data
collected_data = []

# Take live camera input for pose detection
video_path = "C:/Users/KWAME/Downloads//Greetings in ASL _ ASL - American Sign Language.mp4"
cap = cv2.VideoCapture(video_path)

def preprocess_landmarks(landmarks):
    # Convert the landmarks to a flat list of x, y, z coordinates
    data = []
    for landmark in landmarks.landmark:
        data.extend([landmark.x, landmark.y, landmark.z])
    return data

# Main loop for data collection
while True:
    ret, img = cap.read()  # Read a frame from the camera
    img = cv2.resize(img, (600, 400))  # Resize the frame for display

    results = pose.process(img)  # Perform pose detection on the frame

    if results.pose_landmarks:  # If landmarks are detected
        landmarks_data = preprocess_landmarks(results.pose_landmarks)  # Preprocess the landmarks

        # Display the video feed with pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Pose Estimation", img)

        # Wait for key press to label the pose
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('a'), ord('b'), ord('c'),ord('d'),ord('e')]:  # If a valid label key is pressed
            label = chr(key)  # Convert the key to a character label
            collected_data.append({"label": label, "landmarks": landmarks_data})  # Append the labeled data to the list
            print(f"Collected pose {label}")

        if key == ord('q'):  # If 'q' is pressed, exit the loop
            break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows

# Save the collected data to a JSON file
with open(os.path.join(data_dir, 'collected_data.json'), 'w') as f:
    json.dump(collected_data, f)

print(f"Collected {len(collected_data)} samples.")  # Print the number of collected samples
