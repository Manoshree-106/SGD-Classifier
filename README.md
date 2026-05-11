# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2.Calculate the mean of the X -values and the mean of the Y -values.
3.Find the slope m of the line of best fit using the formula.
<img width="350" height="192" alt="Screenshot 2026-05-11 093010" src="https://github.com/user-attachments/assets/760da8dd-bb79-4875-89a8-9a887e9bdf1e" />

4.Compute the y -intercept of the line by using the formula:Screenshot 2026-05-11 093018
<img width="312" height="62" alt="Screenshot 2026-05-11 093018" src="https://github.com/user-attachments/assets/8fa18991-2180-4ce2-b908-600d00f8ae7c" />

5.Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot
 

## Program:
```
/*
Developed by: MANOSHREE N
RegisterNumber: 212225040228
# Implementation of Logistic Regression Using SGD Classifier

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

# Step 2: Create Dataset
data = {
    "CGPA": [5.0, 5.5, 6.0, 6.5, 7.0, 7.2, 7.5, 8.0, 8.5, 9.0],
    "Placement_Status": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display Dataset
print("Dataset:\n")
print(df)

# Step 3: Split Features and Target
X = df[["CGPA"]]                # Independent Variable
y = df["Placement_Status"]      # Dependent Variable

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train SGD Classifier
model = SGDClassifier(
    loss='log_loss',     # Logistic Regression
    max_iter=1000,
    learning_rate='optimal',
    random_state=42
)

model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Probability Predictions
y_prob = model.predict_proba(X_test)

# Step 8: Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_[0])
print("Coefficient (b1):", model.coef_[0][0])

print("\nEvaluation Metrics:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Visualization
plt.figure(figsize=(8,6))

# Scatter Plot
plt.scatter(df["CGPA"], y, color='blue', label="Actual Data")

# Create Smooth Curve
X_range = np.linspace(df["CGPA"].min(), df["CGPA"].max(), 300).reshape(-1,1)

# Scale Input
X_range_scaled = scaler.transform(X_range)

# Predict Probability
y_curve = model.predict_proba(X_range_scaled)[:,1]

# Plot Logistic Curve
plt.plot(
    X_range,
    y_curve,
    color='red',
    linewidth=2,
    label="SGD Logistic Curve"
)

plt.xlabel("CGPA")
plt.ylabel("Placement Probability")
plt.title("Logistic Regression using SGD Classifier")
plt.legend()
plt.grid(True)

plt.show()

# Step 10: Custom Prediction
cgpa = 7.3

# Scale Input
cgpa_scaled = scaler.transform([[cgpa]])

# Predict Class
prediction = model.predict(cgpa_scaled)

# Predict Probability
probability = model.predict_proba(cgpa_scaled)[0][1]

print(f"\nPredicted Placement Status for CGPA {cgpa}:")

if prediction[0] == 1:
    print("Student is Likely to be Placed")
else:
    print("Student is Not Likely to be Placed")

print(f"Placement Probability = {probability:.2f}")
*/
```

## Output:
<img width="642" height="725" alt="Screenshot 2026-05-11 095653" src="https://github.com/user-attachments/assets/e74ccd5f-ea94-4bdd-aae9-d5dee6584a00" />
<img width="952" height="662" alt="Screenshot 2026-05-11 095705" src="https://github.com/user-attachments/assets/f25a3b2a-ae3c-41d5-864f-9b78a17278a0" />
<img width="412" height="73" alt="Screenshot 2026-05-11 095719" src="https://github.com/user-attachments/assets/b6b30523-15d5-445a-bfd1-900311a91906" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
