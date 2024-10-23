# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("C:\\Users\\PC\\OneDrive\\Desktop\\CSB23114\\advertising.csv")



# Check the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Assuming the dataset has columns: 'TV', 'Radio', 'Newspaper', and 'Sales'
# Features (independent variables)
X = df[['TV', 'Radio', 'Newspaper']]  # Advertising spending on different platforms

# Target (dependent variable)
y = df['Sales']  # Sales

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot actual vs predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Scatter plot for Radio vs Sales
plt.figure(figsize=(6, 4))
plt.scatter(df['Radio'], df['Sales'], color='green')
plt.title("Radio Ad Spend vs Sales")
plt.xlabel("Radio Ad Spend")
plt.ylabel("Sales")
plt.show()

# Scatter plot for Newspaper vs Sales
plt.figure(figsize=(6, 4))
plt.scatter(df['Newspaper'], df['Sales'], color='red')
plt.title("Newspaper Ad Spend vs Sales")
plt.xlabel("Newspaper Ad Spend")
plt.ylabel("Sales")
plt.show()

# --- Scatter plots for each feature vs Sales ---
# Scatter plot for TV vs Sales
plt.figure(figsize=(6, 4))
plt.scatter(df['TV'], df['Sales'], color='blue')
plt.title("TV Ad Spend vs Sales")
plt.xlabel("TV Ad Spend")
plt.ylabel("Sales")
plt.show()

# Check for missing values
print(df.isnull().sum())

# --- Correlation Matrix ---
# Calculate the correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)



