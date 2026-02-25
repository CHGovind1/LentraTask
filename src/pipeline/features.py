import pandas as pd
import numpy as np
import yaml

def engineer_features(df, feature_names):
    """YOUR custom features: monthly credits, age limit check, old loans"""
    
    # 1. Monthly crediting check (monthly_payment > threshold = high burden)
    df['monthly_crediting'] = (df['credit_amount'] / (df['duration'] + 1) > 500).astype(int)
    
    # 2. Age limit risk (age > 60 = high risk)
    df['age_limit_risk'] = (df['age'] > 60).astype(int)
    
    # 3. Old loans count (existing_credits + other payment plans)
    df['total_old_loans'] = df['existing_credits'] + (df['other_payment_plans'] > 0).astype(int)
    
    # Update feature names
    new_features = ['monthly_crediting', 'age_limit_risk', 'total_old_loans']
    feature_names_new = feature_names + new_features
    
    return df[feature_names_new], feature_names_new

