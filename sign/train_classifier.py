# Import necessary libraries
import pickle  # For loading and saving data
from sklearnex import patch_sklearn  # For patching scikit-learn for serialization
from sklearnex.ensemble import RandomForestClassifier  # For using the RandomForestClassifier model
from sklearnex.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearnex import metrics  # For evaluating model performance
import numpy as np  # For numerical operations

# Patch scikit-learn to enable serialization of custom objects
patch_sklearn()

# Load the data from a pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])  # Extract features from the data dictionary
labels = np.asarray(data_dict['labels'])  # Extract labels from the data dictionary

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train a RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Use the trained model to make predictions on the testing data
y_predict = model.predict(x_test)

# Define a function to compute the accuracy score
def compute_accuracy(Y_true, Y_pred):
    correctly_predicted = 0
    # Iterate over every label and check it with the true sample
    for true_label, predicted in zip(Y_true, Y_pred):
        if true_label == predicted:
            correctly_predicted += 1
    # Compute the accuracy score
    accuracy_score = correctly_predicted / len(Y_true)
    return accuracy_score

# Calculate the accuracy score
score = compute_accuracy(y_test, y_predict)

# Print the accuracy score
print('Accuracy:', score * 100, '%')

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
