from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

# CORS for Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CSV and model
data_path = os.path.join(os.path.dirname(__file__), '..', '../AI Hakathon.csv')
df = pd.read_csv(data_path)

model_sale = joblib.load("../SALESPRICE_model.pkl")
le = joblib.load("../label_encoder.pkl")

@app.post("/predict")
def predict(data: dict):
    # Find matching data in existing records/db...
    matches = df[
        (df['LOTSIZEACRES'] == data['lot_size']) &
        (df['YEARBUILT'] == data['year_built']) &
        (df['BEDS'] == data['beds']) &
        (df['BATHS'] == data['baths']) &
        (df['PROPERTYTYPE'] == data['property_type'])
    ]

    # Prepare input
    enc_type = le.transform([data['property_type']])[0]
    features = pd.DataFrame([{
        "LOTSIZEACRES": data['lot_size'],
        "YEARBUILT": data['year_built'],
        "BEDS": data['beds'],
        "BATHS": data['baths'],
        "PROPERTYTYPE_ENC": enc_type
    }])

    # Price prediction through our trained model
    predicted_price = model_sale.predict(features)[0]

    # Existing record ki average
    if not matches.empty:
        avg_sale = matches['SALESPRICE'].mean()
        avg_estimate = matches['ESTIMATEDVALUE'].mean()
        avg_conf = matches['CONFIDENCESCORE'].mean()

        # Confidence calculation
        confidence = max(0, 100 - abs(predicted_price - avg_sale) / avg_sale * 100)

        return {
            "predicted_sale_price": round(predicted_price),
            "confidence_value": round(confidence),
            "actual_sale_price": round(avg_sale),
            "actual_estimated_value": round(avg_estimate),
            "actual_confidence_value": round(avg_conf),
            "error_vs_sale": round((predicted_price - avg_sale) / avg_sale * 100, 2),
            "error_vs_estimate": round((predicted_price - avg_estimate) / avg_estimate * 100, 2),
        }
    else:
        return {
            "predicted_sale_price": round(predicted_price),
            "confidence_value": 100,
            "actual_sale_price": None,
            "actual_estimated_value": None,
            "actual_confidence_value": 0,            
            "error_vs_sale": None,
            "error_vs_estimate": None
        }
