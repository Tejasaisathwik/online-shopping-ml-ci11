import unittest
import app

class TestShoppingAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "healthy")

    def test_prediction_endpoint(self):
        sample = {
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
        }

        response = self.client.post("/predict", json=sample)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json)

if __name__ == "__main__":
    unittest.main()
