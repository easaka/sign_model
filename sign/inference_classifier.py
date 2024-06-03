# Import necessary libraries
import pickle  # For loading the model
import cv2  # For accessing the webcam and image processing
import mediapipe as mp  # For hand tracking and landmark detection
import numpy as np  # For numerical operations

# Load the trained model from a pickle file
model_dict = pickle.load(open('./sign/model.p', 'rb'))
model = model_dict['model']

# Open the webcam to capture video
pic="C:/Users/KWAME/OneDrive/Documents/personal/project/sign-language-detector-python/sign/data/0/A0 - Copy - Copy - Copy.jpg"
cap = cv2.VideoCapture(0)

# Initialize the MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define a dictionary mapping class indices to corresponding labels
labels_dict = {0: 'A', 1: 'B', 2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H'}

# Start an infinite loop to continuously capture and process video frames
while True:
    # Initialize variables to store hand landmark data
    data_aux = []
    x_ = []
    y_ = []

    # Read a frame from the webcam
    ret, frame = cap.read()

    # Get the dimensions of the frame (height, width, channels)
    H, W, _ = frame.shape

    # Convert the frame from BGR to RGB color space (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the MediaPipe Hands module to detect hand landmarks
    results = hands.process(frame_rgb)

    # If hand landmarks are detected in the frame
    if results.multi_hand_landmarks:
        # Iterate over each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,  # Image to draw on
                hand_landmarks,  # Hand landmarks data
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Iterate over each detected hand again to extract landmark coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                # Get the normalized coordinates of each landmark
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Store the x and y coordinates in separate lists
                x_.append(x)
                y_.append(y)

            # Calculate the relative coordinates of each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate the bounding box coordinates of the hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make a prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])

        # Map the predicted class index to its corresponding label
        predicted_character = labels_dict[int(prediction[0])]

        # Draw a bounding box around the hand and display the predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the processed frame with annotations
    cv2.imshow('frame', frame)

    # Check for key press events
    key = cv2.waitKey(1)

    # If the 'q' key is pressed, exit the loop and close the program
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
