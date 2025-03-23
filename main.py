import xgboost as xgb
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Load the trained XGBoost model from JSON
model = xgb.Booster()
model.load_model("updated_xgb_model.json")  # Load model from JSON file

# ✅ Load the scaler
scaler = joblib.load("scaler.pkl")  # Using joblib to load the scaler

# ✅ Define Input Schema using Pydantic
class TransactionInput(BaseModel):
    transaction_id: str
    transaction_amount: float
    payer_email: str
    payee_ip: str

@app.post("/predict/")
def predict_fraud(data: TransactionInput):
    """ Predict fraud using the trained XGBoost model in JSON format """

    try:
        # ✅ Rename fields to match model expectations
        features_dict = {
            "payer_email_anonymous": hash(data.payer_email) % (10**6),
            "payee_id_anonymous": hash(data.payee_ip) % (10**6),
            "transaction_amount": data.transaction_amount
        }

        # ✅ Convert dictionary to XGBoost DMatrix
        dmatrix = xgb.DMatrix(np.array([[
                                         features_dict["payer_email_anonymous"],
                                         features_dict["payee_id_anonymous"],features_dict["transaction_amount"],]]),
                              feature_names=["payer_email_anonymous", "payee_id_anonymous","transaction_amount",])

        # ✅ Predict fraud
        fraud_probability = model.predict(dmatrix)[0]

        # ✅ Convert probability to binary fraud prediction
        is_fraud = int(fraud_probability > 0.5)

        return {
            "transaction_id": data.transaction_id,
            "is_fraud": is_fraud
        }

    except Exception as e:
        return {"error": str(e)}
