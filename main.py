from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_fraud

app = FastAPI()

# Input schema
class TransactionInput(BaseModel):
    transaction_id: str
    transaction_date: str
    transaction_amount: float
    transaction_channel: str
    transaction_payment_mode: int
    payment_gateway_bank: int
    payer_email: str
    payer_mobile: str
    payer_browser: int
    payee_id: str
    payee_ip: str

# Output schema
class FraudPredictionOutput(BaseModel):
    transaction_id: str
    is_fraud: int

@app.post("/predict", response_model=FraudPredictionOutput)
def predict(transaction: TransactionInput):
    is_fraud = predict_fraud(transaction.dict())
    return {"transaction_id": transaction.transaction_id, "is_fraud": is_fraud}

