# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_target = pd.Series(data=iris.target, name='species')

# Concatenate data and target to create the complete dataset
iris_df = pd.concat([iris_data, iris_target], axis=1)

# Define input features (X) and target variable (y)
X = iris_data[['sepal length (cm)', 'petal length (cm)']]
y = iris_target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a K-Nearest Neighbors classifier with k=3
knn_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model on the training data
knn_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# You can now use the model to predict the species of flowers for new data
new_data = pd.DataFrame({'sepal length (cm)': [5.1, 6.2],
                         'petal length (cm)': [1.4, 4.5]})

new_predictions = knn_model.predict(new_data)
print("New Data Predictions:")
for i, prediction in enumerate(new_predictions):
    print(f"Data {i+1}: Predicted Species = {iris.target_names[prediction]}")