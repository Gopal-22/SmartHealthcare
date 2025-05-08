import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Load dataset
try:
    lung_cancer_df = pd.read_csv('lung_cancer_data.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("🚨 Error: 'lung_cancer_data.csv' file not found. Please ensure the file is in the correct directory.")
    exit()

# Convert column names to lowercase and strip spaces
lung_cancer_df.columns = lung_cancer_df.columns.str.lower().str.strip()

# Print columns before dropping
print("Dataset Columns Before Dropping:", lung_cancer_df.columns)

# Define target column
target_column = 'lung_cancer'

# Check if target column exists before proceeding
if target_column in lung_cancer_df.columns:
    X = lung_cancer_df.drop(columns=[target_column])
    y = lung_cancer_df[target_column]
    print("✅ Target column found and removed.")
else:
    print(f"🚨 Error: '{target_column}' not found in dataset columns.")
    print("Columns in dataset:", lung_cancer_df.columns)
    exit()

# Encode non-numeric columns
for column in X.columns:
    if X[column].dtype == 'object':
        print(f"Encoding non-numeric column: {column}")
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Normalize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

try:
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("✅ Hyperparameter tuning completed.")
except Exception as e:
    print(f"🚨 Error during hyperparameter tuning: {str(e)}")
    exit()

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Train the best model
try:
    best_model.fit(X_train, y_train)
    print("✅ Model training completed.")
except Exception as e:
    print(f"🚨 Error during model training: {str(e)}")
    exit()

# Evaluate the model
try:
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
except Exception as e:
    print(f"🚨 Error during model evaluation: {str(e)}")
    exit()

# Save the trained model
try:
    joblib.dump(best_model, 'lung_cancer_random_forest.sav')
    print("✅ Model trained and saved as 'lung_cancer_random_forest.sav'.")
except Exception as e:
    print(f"🚨 Error saving the model: {str(e)}")