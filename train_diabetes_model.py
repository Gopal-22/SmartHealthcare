import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset (Using Pima Indians Diabetes Dataset as an example)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Read the dataset
df = pd.read_csv(url, names=columns)

# Handle missing or zero values (replace zeros with NaN and fill with column mean)
columns_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[columns_to_check] = df[columns_to_check].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Split features and target variable
X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]  # Target (0 = No Diabetes, 1 = Diabetes)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
with open('diabetes_pred.sav', 'wb') as file:
    pickle.dump(best_model, file)

print("✅ Model saved successfully as 'diabetes_pred.sav'.")