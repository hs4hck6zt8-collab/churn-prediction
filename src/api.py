from fastapi import FastAPI
from plotly.graph_objs.indicator.gauge import threshold
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import shap

# Загрузка модели
artifact = joblib.load("models/churn_model.pkl")
model = artifact["model"]
threshold = artifact["threshold"]

preprocessor = model.named_steps["preprocessor"]
lgbm_model = model.named_steps["classifier"].estimators_[0]
explainer = shap.TreeExplainer(lgbm_model)

app = FastAPI()

# Схема запроса
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup:str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure_segment: str
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    monthly_charges_per_service: float
    has_multiple_contracts: bool

CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_segment'
]
NUM_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'monthly_charges_per_service', 'has_multiple_contracts'
]

# /predict
@app.post("/predict")
def predict(customer: CustomerData):
    df = pd.DataFrame([customer.dict()])
    X = df[CAT_COLS + NUM_COLS]
    proba = model.predict_proba(X)[0][1]
    churn = int(proba >= threshold)
    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": churn,
        "threshold_used": threshold,
    }

# /explain
@app.post("/explain")
def explain(customer: CustomerData):
    df = pd.DataFrame([customer.dict()])
    X = df[CAT_COLS + NUM_COLS]
    X_transformed = preprocessor.transform(X)
    shap_raw = explainer.shap_values(X_transformed)
    if isinstance(shap_raw, list):
        shap_vals = shap_raw[1][0]
    else:
        shap_vals = shap_raw[0]
    feature_names = CAT_COLS + NUM_COLS
    top =sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    return {
        "top_factors": [
            {"feature": f, "shap_value": round(float(v), 4)}
            for f, v in top
        ]
    }

# /health
@app.get("/health")
def health():
    return {"status": "ok"}