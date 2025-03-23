import pickle
import xgboost as xgb

# Load the model from pickle file
with open("xgboost_fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# Check if it's an XGBoost model
if isinstance(model, xgb.Booster):
    print("Model is already in XGBoost Booster format.")
elif isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
    print("Converting sklearn model to Booster...")
    model = model.get_booster()  # Convert to Booster format
else:
    raise ValueError("Unsupported model type:", type(model))
# Save in correct format
model.save_model("updated_xgb_model.json")
print("Model saved successfully as updated_xgb_model.json")
