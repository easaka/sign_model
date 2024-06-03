import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle

# Load the collected data
data_dir = "./pose/sign_data"
with open(os.path.join(data_dir, 'collected_data.json'), 'r') as f:
    collected_data = json.load(f)

# Extract landmarks and labels
X = np.array([sample['landmarks'] for sample in collected_data])
y = np.array([sample['label'] for sample in collected_data])

# Check the shape of X to verify the number of features
print(f"Shape of X: {X.shape}")

# Normalize the landmarks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained classifier and scaler
with open('./pose/sign_pose_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('./pose/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
