import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import logging

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def preprocess_data(df):
    """Clean, encode, and split data"""
    config = load_config()
    
    # Remap target: 1=Good(0), 2=Bad(1)
    df['class'] = df['class'] - 1
    
    # Identify categorical columns (all non-numeric)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [col for col in cat_cols if col != 'class']
    
    # Label encode categoricals
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Split
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], 
        random_state=config["model"]["random_state"], 
        stratify=y
    )
    
    # Scale numerics
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler, 'le_dict': le_dict,
        'feature_names': X.columns.tolist()
    }

