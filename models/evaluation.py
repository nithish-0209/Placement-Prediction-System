"""
M5: Model Evaluation, Selection, and Calibration
Comprehensive evaluation suite:
- Stratified K-Fold CV & Classification metrics (Acc, Prec, Rec, F1, ROC-AUC, PR-AUC)
- Confusion Matrix & Imbalance Analysis
- Regression Metrics (RMSE, MAE, MAPE, R2) for Salary Prediction
- Probability Calibration (Platt Scaling vs Isotonic Regression, Reliability Diagrams, Brier Score)
- McNemar's Statistical Significance Test for Classifier Comparison
- Learning Curves generation
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

def evaluate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes rigorous classification metrics.
    Highlights why Accuracy lies under class imbalance (e.g. 74% negative class).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'specificity': round(float(specificity), 4),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }

    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        metrics['roc_auc'] = round(float(roc_auc), 4)
        metrics['pr_auc'] = round(float(pr_auc), 4)
        metrics['brier_score'] = round(float(brier), 4)

    return metrics

def evaluate_regression_metrics(y_true, y_pred):
    """
    Regression evaluation metrics:
    - RMSE: sensitive to large errors (penalizes outliers quadratically)
    - MAE: robust to outliers (linear penalty)
    - MAPE: percentage error scale
    - R2: proportion of variance explained
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Avoid div-by-zero in MAPE
    mask = y_true > 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
        
    r2 = r2_score(y_true, y_pred)

    return {
        'rmse': round(float(rmse), 4),
        'mae': round(float(mae), 4),
        'mape': round(float(mape), 2),
        'r2_score': round(float(r2), 4)
    }

def compute_calibration_diagnostics(y_true, y_prob, n_bins=10):
    """
    Probability Calibration Reliability Curve:
    Plots mean predicted probability vs true fraction of positives.
    A well-calibrated model follows the diagonal y = x.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        'prob_pred': [float(p) for p in prob_pred],
        'prob_true': [float(p) for p in prob_true],
        'brier_score': round(float(brier), 4)
    }

def calibrate_model(base_model, X_train, y_train, method='sigmoid'):
    """
    Calibrates predicted probabilities:
    - method='sigmoid': Platt Scaling (fits a logistic sigmoid over raw margins)
    - method='isotonic': Non-parametric isotonic regression (step function, best for large datasets)
    """
    calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated

def run_mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's Statistical Test for paired nominal comparison:
    Determines if the difference in error rates between Model A and Model B is statistically significant.
    b: Model A correct, Model B incorrect
    c: Model A incorrect, Model B correct
    chi2_stat = (|b - c| - 1)^2 / (b + c)
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency Table
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right

    if (b + c) == 0:
        return {'chi2_stat': 0.0, 'p_value': 1.0, 'significant': False, 'table': {'b': 0, 'c': 0}}

    # Edwards continuity correction
    chi2_stat = (abs(b - c) - 1.0)**2 / (b + c)
    p_val = chi2.sf(chi2_stat, df=1)

    return {
        'chi2_stat': round(float(chi2_stat), 4),
        'p_value': float(p_val),
        'significant': bool(p_val < 0.05),
        'contingency_table': {'b_A_right_B_wrong': int(b), 'c_A_wrong_B_right': int(c)}
    }

def compute_learning_curves(model, X, y, cv=3, n_points=5):
    """
    Diagnoses bias vs variance from learning curve trajectory.
    High bias (underfitting): both train and validation curves converge at low score.
    High variance (overfitting): wide gap between train and validation scores.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X[:5000], y[:5000], cv=cv, scoring='f1',
        train_sizes=np.linspace(0.2, 1.0, n_points),
        n_jobs=-1, random_state=42
    )

    return {
        'train_sizes': [int(s) for s in train_sizes],
        'train_scores_mean': [float(s) for s in np.mean(train_scores, axis=1)],
        'val_scores_mean': [float(s) for s in np.mean(val_scores, axis=1)]
    }
