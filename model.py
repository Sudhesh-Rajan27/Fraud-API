import torch
import pickle
import numpy as np

# Load the trained GNN fraud detection model
model_path = "gnn_fraud_model.pt"
scaler_path = "scaler.pkl"

# Load the model
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Load the scaler for normalization
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

def preprocess_input(data):
    """Convert JSON input to a NumPy array & apply scaling."""
    features = [
        data["transaction_amount"],
        data["transaction_payment_mode"],
        data["payment_gateway_bank"],
        data["payer_browser"],
    ]
    features = np.array(features).reshape(1, -1)
    return torch.tensor(scaler.transform(features), dtype=torch.float32)

def predict_fraud(data):
    """Make a fraud prediction from the input JSON."""
    input_tensor = preprocess_input(data)
    with torch.no_grad():
        prediction = model(input_tensor)
    return int(prediction.item() > 0.5)  # Convert probability to 0/1
