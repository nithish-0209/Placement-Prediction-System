"""
M3: Supervised Learning — Tree-Based Models
Implements and benchmarks tree-based architectures:
- Decision Trees with Gini vs Entropy and Cost-Complexity Pruning (CCP)
- Random Forests with Bagging, Feature Subsampling, and Out-of-Bag (OOB) Error
- Gradient Boosting / HistGradientBoosting / XGBoost (Histogram-based, leaf-wise growth)
- Feature Importance: MDI (Gini Importance) vs Permutation Importance
- Why Trees routinely beat Neural Networks on Tabular Data
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

def train_decision_tree(X_train, y_train, criterion='gini', max_depth=6, ccp_alpha=0.0, random_state=42):
    """
    Decision Tree with pre-pruning (max_depth, min_samples_leaf)
    and post-pruning (CCP: Cost-Complexity Pruning).
    Splitting criteria: Gini Impurity (1 - sum(p_i^2)) or Entropy (-sum(p_i * log2(p_i))).
    """
    dt = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )
    dt.fit(X_train, y_train)
    return dt

def compute_ccp_path(X_train, y_train, max_samples=5000):
    """Computes effective alphas for Cost-Complexity Pruning post-pruning path."""
    X_sub = X_train[:max_samples]
    y_sub = y_train[:max_samples]
    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X_sub, y_sub)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    # Return a subsample of alphas for visualization
    indices = np.linspace(0, len(ccp_alphas) - 1, min(10, len(ccp_alphas))).astype(int)
    return {
        'alphas': [float(a) for a in ccp_alphas[indices]],
        'impurities': [float(imp) for imp in impurities[indices]]
    }

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=12, random_state=42):
    """
    Random Forest:
    - Bagging: bootstrap sampling of training instances (variance reduction)
    - Feature subsampling: m = sqrt(d) features considered at each split (decorrelating trees)
    - Out-Of-Bag (OOB) score: validates generalization without separate validation set
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        oob_score=True,
        n_jobs=-1,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf

def train_gradient_boosting(X_train, y_train, learning_rate=0.1, max_iter=100, random_state=42):
    """
    Histogram-Based Gradient Boosting (industrial workhorse similar to LightGBM/XGBoost):
    - Histogram-based binning: continuous features binned into 256 integer bins (massive speedup)
    - Leaf-wise growth
    - Functional gradient descent: each tree fits the negative gradient (pseudo-residuals) of loss
    """
    hgb = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=8,
        l2_regularization=1.0,
        random_state=random_state
    )
    hgb.fit(X_train, y_train)
    return hgb

def extract_tree_feature_importance(model, feature_names, X_val=None, y_val=None):
    """
    Extracts Gini Importance (Mean Decrease in Impurity - MDI)
    and Permutation Importance (unbiased metric evaluated on validation data).
    """
    names = feature_names[:30]
    result = {'features': names}

    # 1. Gini Importance (MDI) if model has feature_importances_
    if hasattr(model, 'feature_importances_'):
        mdi = model.feature_importances_[:len(names)]
        result['gini_importance'] = [float(x) for x in mdi]
    else:
        result['gini_importance'] = []

    # 2. Permutation Importance (evaluated on validation set)
    if X_val is not None and y_val is not None:
        perm_res = permutation_importance(model, X_val[:1000], y_val[:1000], n_repeats=5, random_state=42)
        result['permutation_importance'] = [float(x) for x in perm_res.importances_mean[:len(names)]]

    return result

WHY_TREES_BEAT_NN_ON_TABULAR = [
    {
        "factor": "Step-Function Decision Boundaries",
        "explanation": "Tabular data features have sharp, disjoint thresholds (e.g. CGPA >= 7.5 or Backlogs == 0). Trees represent orthogonal axis-aligned step functions naturally, whereas neural networks struggle with high-frequency step functions due to smooth activation biases."
    },
    {
        "factor": "Invariance to Monotone Transformations",
        "explanation": "Decision trees only care about feature ordering, not numeric scale or distribution skew. Neural nets are hypersensitive to outliers, scale differences, and heavy-tailed distributions."
    },
    {
        "factor": "Mixed Data Types & Cardinality",
        "explanation": "Tabular datasets blend continuous, high-cardinality nominal, and ordinal features. Tree splitters evaluate each column independently without suffering from the curse of dimensionality across heterogeneous representations."
    },
    {
        "factor": "Native Handling of Missing Values",
        "explanation": "Modern tree algorithms (LightGBM/XGBoost/HistGradientBoosting) assign missing values to optimal split directions dynamically without arbitrary imputation distortion."
    }
]
