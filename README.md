# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and create a dataset containing the number of hours studied and marks scored.
2. Separate the independent variable Hours Studied (X) and dependent variable Marks Scored (Y).
3. Split the dataset into training and testing data and train the Simple Linear Regression model.
4. Predict the marks using the trained model and evaluate the model using the R² score.
5. Plot the actual data points and the regression line.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: UDHAYA DHARSHINI.T 
RegisterNumber:  212225230288
*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
marks = np.array([20, 25, 35, 40, 50, 55, 65, 70, 80, 90])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    hours, marks, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict marks
y_pred = model.predict(X_test)

# Display coefficient and intercept
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Display R2 score
print("R2 Score:", r2_score(y_test, y_pred))

# Predict marks for a student studying 7 hours
hours_studied = [[7]]
predicted_marks = model.predict(hours_studied)

print("Predicted marks for 7 hours of study:",
      predicted_marks[0])

# Plot the data and regression line
plt.scatter(hours, marks, color="blue", label="Actual Marks")
plt.plot(hours, model.predict(hours),
         color="red", label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression - Marks Prediction")
plt.legend()
plt.show()
```

## Output:
<img width="907" height="762" alt="Screenshot 2026-08-21 142450" src="https://github.com/user-attachments/assets/d5590ae3-c61c-400e-a5a1-1f2d50897727" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
