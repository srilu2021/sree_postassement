import joblib
import pandas as pd

# Load the trained model and label encoders
model = joblib.load("app/model.pkl")
label_encoders = joblib.load("app/label_encoders.pkl")

# Preprocess function for prediction
def preprocess_input(input_data):
    input_data['Warehouse_block'] = label_encoders['Warehouse_block'].transform(input_data['Warehouse_block'])
    input_data['Mode_of_Shipment'] = label_encoders['Mode_of_Shipment'].transform(input_data['Mode_of_Shipment'])
    input_data['Product_importance'] = label_encoders['Product_importance'].transform(input_data['Product_importance'])
    input_data['Gender'] = label_encoders['Gender'].transform(input_data['Gender'])
    return input_data

# Predict function
def predict(features):
    features = preprocess_input(features)
    prediction = model.predict(features)
    return prediction
