# ===============================
# Task-15: End-to-End ML Pipeline
# ===============================

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

# 1️⃣ Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2️⃣ Identify Numerical Features
numerical_features = X.columns.tolist()

# 3️⃣ Preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features)
    ]
)

# 4️⃣ Create ML Pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Train Model
pipeline.fit(X_train, y_train)

# 7️⃣ Predictions
y_pred = pipeline.predict(X_test)

# 8️⃣ Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save classification report
os.makedirs("outputs", exist_ok=True)
with open("outputs/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

# 9️⃣ Save Pipeline Model
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/breast_cancer_pipeline.pkl")

print("\nPipeline saved successfully!")
