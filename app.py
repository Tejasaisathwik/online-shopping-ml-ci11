
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("artifacts/shopping_model.pkl")

FEATURES = [
    "Age", "Gender", "MonthlyIncome", "WebsiteVisits",
    "PagesViewed", "TimeSpentMinutes", "PreviousPurchases",
    "CartItems", "DiscountUsed", "DeviceType", "Membership",
    "CustomerRating", "PurchaseAmount"
]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({"error": "Missing fields", "fields": missing}), 400

    try:
        input_df = pd.DataFrame([
            {feature: data[feature] for feature in FEATURES}
        ])
        prediction = int(model.predict(input_df)[0])

        return jsonify({
            "prediction": prediction,
            "label": "Purchased" if prediction == 1 else "Not Purchased"
        })
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
