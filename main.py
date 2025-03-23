import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Load the trained XGBoost model
with open("xgboost_fraud_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# ✅ Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# ✅ FastAPI App
app = FastAPI()

# ✅ Define Input Schema using Pydantic
class TransactionInput(BaseModel):
    transaction_id: str
    transaction_amount: float
    payer_email: str
    payee_ip: str

@app.post("/predict/")
def predict_fraud(data: TransactionInput):
    """ Predict fraud using the trained XGBoost model """

    # ✅ Scale transaction_amount
    scaled_amount = scaler.transform(np.array([[data.transaction_amount]]))  # Ensure 2D array

    # ✅ Prepare input features (assuming model expects these three)
    features = np.array([
        scaled_amount[0, 0],  # Extract single value after scaling
        hash(data.payer_email) % (10**6),  # Convert email to numeric hash
        hash(data.payee_ip) % (10**6)  # Convert IP to numeric hash
    ]).reshape(1, -1)

    # ✅ Predict fraud
    is_fraud = int(model.predict(features)[0])  # Convert NumPy type to native Python int

    return {"transaction_id": data.transaction_id, "is_fraud": is_fraud}
