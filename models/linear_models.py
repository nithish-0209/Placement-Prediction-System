"""
M2: Supervised Learning — Linear Models at Depth
Implements and analyzes linear models:
- Ridge (L2), Lasso (L1 with sparsity induction), and ElasticNet Logistic Regression
- Normal Equations vs Gradient Descent comparison for Linear Regression
- Hyperplane decision boundary, Sigmoid function, Cross-Entropy loss
- Model coefficient interpretation & Odds Ratios
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error, r2_score

def sigmoid(z):
    """Computes the standard logistic sigmoid: 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def cross_entropy_loss(y_true, y_prob):
    """Computes binary cross-entropy loss: -[y*log(p) + (1-y)*log(1-p)]."""
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

class NormalEquationSolver:
    """
    Closed-Form Normal Equation: theta = (X^T X)^(-1) X^T y
    Compared with Gradient Descent (SGDRegressor) to highlight computational complexity:
    O(n * d^2 + d^3) for matrix inversion vs O(k * n * d) for GD.
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.theta = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_b = X
        
        # theta = (X_b^T X_b)^(-1) X_b^T y using pseudo-inverse for numerical stability
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_b = X
        return X_b @ self.theta

def train_ridge_logistic(X_train, y_train, C=1.0, random_state=42):
    """
    Ridge (L2) Regularized Logistic Regression:
    Loss = CrossEntropy + (1 / 2C) * ||w||_2^2
    Prevents extreme weights, shrinks coefficients smoothly towards zero without setting them to 0.
    """
    clf = LogisticRegression(
        penalty='l2',
        C=C,
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf

def train_lasso_logistic(X_train, y_train, C=0.5, random_state=42):
    """
    Lasso (L1) Regularized Logistic Regression:
    Loss = CrossEntropy + (1 / C) * ||w||_1
    Geometric property: Diamond L1 contour contacts loss ellipses at corners,
    driving uninformative coefficients exactly to ZERO (inducing sparsity).
    """
    clf = LogisticRegression(
        penalty='l1',
        C=C,
        solver='saga',
        max_iter=1000,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf

def train_elastic_net_logistic(X_train, y_train, C=1.0, l1_ratio=0.5, random_state=42):
    """
    Elastic Net Logistic Regression:
    Combines L1 (sparsity) and L2 (grouping effect for correlated features).
    """
    clf = LogisticRegression(
        penalty='elasticnet',
        C=C,
        l1_ratio=l1_ratio,
        solver='saga',
        max_iter=1000,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf

def interpret_coefficients(clf, feature_names):
    """
    Extracts weights, computes Odds Ratios = exp(beta),
    identifies zeroed coefficients (sparsity from Lasso), and ranks importance.
    """
    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)
    
    interp_df = pd.DataFrame({
        'feature': feature_names[:len(coefs)],
        'coefficient': coefs,
        'odds_ratio': odds_ratios,
        'is_sparse_zero': np.isclose(coefs, 0.0, atol=1e-4)
    })
    
    interp_df['absolute_impact'] = np.abs(interp_df['coefficient'])
    interp_df = interp_df.sort_values(by='absolute_impact', ascending=False).reset_index(drop=True)
    return interp_df

def compare_linear_regression_methods(X_train, y_train, X_test, y_test):
    """Compares Closed-Form Normal Equation vs Gradient Descent (SGD) on Salary Prediction."""
    import time

    # 1. Closed-form Normal Equation
    t0 = time.time()
    normal_eq = NormalEquationSolver()
    normal_eq.fit(X_train, y_train)
    time_normal = time.time() - t0
    preds_normal = normal_eq.predict(X_test)
    rmse_normal = np.sqrt(mean_squared_error(y_test, preds_normal))
    r2_normal = r2_score(y_test, preds_normal)

    # 2. Gradient Descent (SGDRegressor)
    t0 = time.time()
    sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    sgd.fit(X_train, y_train)
    time_sgd = time.time() - t0
    preds_sgd = sgd.predict(X_test)
    rmse_sgd = np.sqrt(mean_squared_error(y_test, preds_sgd))
    r2_sgd = r2_score(y_test, preds_sgd)

    return {
        'normal_equation': {
            'rmse': float(rmse_normal), 'r2': float(r2_normal), 'training_time_sec': float(time_normal),
            'method': 'Analytical (X^T X)^-1 X^T y', 'complexity': 'O(n * d^2 + d^3)'
        },
        'gradient_descent_sgd': {
            'rmse': float(rmse_sgd), 'r2': float(r2_sgd), 'training_time_sec': float(time_sgd),
            'method': 'Iterative Stochastic Gradient Descent', 'complexity': 'O(epochs * n * d)'
        }
    }
