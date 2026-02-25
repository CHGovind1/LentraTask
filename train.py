import mlflow
import mlflow.sklearn
import logging
import pandas as pd
import os
from src.pipeline.ingest import ingest_data
from src.pipeline.preprocess import preprocess_data
from src.pipeline.features import engineer_features
from src.pipeline.train import train_model
from src.pipeline.evaluate import evaluate_model
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("german-credit-risk")
    
    with mlflow.start_run(run_name="docker-baseline"):
        logger.info(f"Using MLflow: {tracking_uri}")
        
        df = ingest_data()
        processed = preprocess_data(df)
        
        X_train_eng, _ = engineer_features(processed['X_train'].copy(), processed['feature_names'])
        X_test_eng, _ = engineer_features(processed['X_test'].copy(), processed['feature_names'])
        
        model = train_model(X_train_eng, processed['y_train'])
        metrics = evaluate_model(model, X_test_eng, processed['y_test'])
        
        mlflow.log_params({"test_size": 0.2, "random_state": 42})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"✅ AUC-ROC: {metrics['auc_roc']:.3f}")

if __name__ == "__main__":
    main()
