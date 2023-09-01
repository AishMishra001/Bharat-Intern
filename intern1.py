import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a sample dataset
np.random.seed(42)
num_samples = 100
X = np.random.randint(100, 2000, size=num_samples)  # Square footage
y = 100 + 5 * X + np.random.normal(0, 200, size=num_samples)  # Simulated house prices

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Predicted')
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.title('Linear Regression: House Price Prediction')
plt.legend()
plt.show()
