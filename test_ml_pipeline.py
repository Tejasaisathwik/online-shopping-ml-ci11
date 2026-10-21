import os
import json
import joblib
import pandas as pd

def test_model_files_exist():
    assert os.path.exists("artifacts/shopping_model.pkl")
    assert os.path.exists("artifacts/metrics.json")

def test_metrics_are_valid():
    with open("artifacts/metrics.json") as file:
        metrics = json.load(file)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1

def test_model_can_predict():
    model = joblib.load("artifacts/shopping_model.pkl")

    sample = pd.DataFrame([{
        "Age": 25,
        "Gender": "Male",
        "MonthlyIncome": 30000,
        "WebsiteVisits": 5,
        "PagesViewed": 8,
        "TimeSpentMinutes": 10,
        "PreviousPurchases": 2,
        "CartItems": 1,
        "DiscountUsed": "Yes",
        "DeviceType": "Mobile",
        "Membership": "Regular",
        "CustomerRating": 4,
        "PurchaseAmount": 1500
    }])

    prediction = model.predict(sample)
    assert len(prediction) == 1
