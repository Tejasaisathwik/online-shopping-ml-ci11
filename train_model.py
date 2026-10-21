
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "data/online_shopping_purchase_prediction.csv"

os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Purchased"])
y = df["Purchased"]

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    "model": "Random Forest"
}

joblib.dump(model, "artifacts/shopping_model.pkl")

with open("artifacts/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed successfully.")
print("Accuracy:", metrics["accuracy"])
print("F1 Score:", metrics["f1_score"])
print("Saved: artifacts/shopping_model.pkl")
