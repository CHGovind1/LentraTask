from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import logging

def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    logging.info(f"Metrics: {metrics}")
    return metrics
 
