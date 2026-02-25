from sklearn.ensemble import RandomForestClassifier
import yaml
import logging
import mlflow
import mlflow.sklearn

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model(X_train, y_train):
    """Train RandomForest model"""
    config = load_config()
    
    model = RandomForestClassifier(**config["model"]["params"])
    model.fit(X_train, y_train)
    
    logging.info("Model trained")
    return model
