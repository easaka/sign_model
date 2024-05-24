import cv2
import mediapipe as mp
import numpy as np
import os
import json

# Initialize mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create a directory to save landmark data
data_dir = "./sign_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define a list to store the collected data
collected_data = []
labels = ["Absent", "Again", "ASl", "Bathroom", "Due", "Favorite", "Go-to", "Homework", "learn", "Movie", "Need", "No", "Please", "Practice", "School", "Sign", "Slow-down", "thank you", "Today", "Yes"]
label_index = 0

# Take live camera input for pose detection
video_path = "C:/Users/KWAME/Downloads//American Sign Language.mp4"
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
    if not ret:
        break
    img = cv2.resize(img, (600, 400))  # Resize the frame for display

    results = pose.process(img)  # Perform pose detection on the frame

    if results.pose_landmarks:  # If landmarks are detected
        landmarks_data = preprocess_landmarks(results.pose_landmarks)  # Preprocess the landmarks

        # Display the video feed with pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Pose Estimation", img)

        # Wait for key press to label the pose
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('s') and label_index < len(labels):  # Press 's' to label the current pose
            label = labels[label_index]  # Get the current label from the array
            collected_data.append({"label": label, "landmarks": landmarks_data})  # Append the labeled data to the list
            print(f"Collected pose {label}")
            # label_index += 1  # Increment the label index after labeling
            if label_index >= len(labels):
                print("All labels have been assigned.")
                break
        elif key == ord('n'):  # Press 'n' to go to the next pose
            if label_index < len(labels):
                label_index += 1  # Increment the label index
                print(f"Next label: {labels[label_index]}")
            else:
                print("No more signs")
                break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows

# Save the collected data to a JSON file
with open(os.path.join(data_dir, 'collected_data.json'), 'w') as f:
    json.dump(collected_data, f)

print(f"Collected {len(collected_data)} samples.")  # Print the number of collected samples
