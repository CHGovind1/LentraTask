from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow.sklearn
import yaml
import joblib
import logging
import os
import minio
from minio import Minio

app = FastAPI(title="Credit Risk Prediction API")

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global model and scaler
model = None
feature_names = None

class PredictionRequest(BaseModel):
    records: List[dict]

@app.on_event("startup")
async def load_model():
    global model, feature_names
    
    # Connect to MinIO
    minio_client = Minio(
        config["minio"]["endpoint"],
        access_key=config["minio"]["access_key"],
        secret_key=config["minio"]["secret_key"],
        secure=False
    )
    
    # Get latest model from MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_uri = "runs:/<latest-run-id>/model"  # Replace with actual run ID after training
    
    # Load model from MinIO artifact
    model = mlflow.sklearn.load_model(model_uri)
    logging.info("Model loaded successfully")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    global model, feature_names
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(request.records)
        
        # Feature engineering (same as training)
        df_eng = engineer_features(df, feature_names)
        
        # Predict
        predictions = model.predict(df_eng)
        probabilities = model.predict_proba(df_eng)[:, 1]
        
        return {
            "predictions": predictions.tolist(),  # 0=Good, 1=Bad
            "probabilities": probabilities.tolist(),
            "records_processed": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def engineer_features(df, feature_names):
    """Same feature engineering as training pipeline"""
    # Your custom features
    df['monthly_crediting'] = (df['credit_amount'] / (df['duration'] + 1) > 500).astype(int)
    df['age_limit_risk'] = (df['age'] > 60).astype(int)
    df['total_old_loans'] = df['existing_credits'] + (df['other_payment_plans'] > 0).astype(int)
    
    return df[feature_names + ['monthly_crediting', 'age_limit_risk', 'total_old_loans']]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 
