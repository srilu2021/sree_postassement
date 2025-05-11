from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    payload = {
        "Warehouse_block": "A",
        "Mode_of_Shipment": "Flight",
        "Customer_care_calls": 3,
        "Customer_rating": 4,
        "Cost_of_the_Product": 200.0,
        "Prior_purchases": 2,
        "Product_importance": "low",
        "Gender": "M",
        "Discount_offered": 10.0,
        "Weight_in_gms": 3000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "Reached_on_Time_Y_N" in response.json()
