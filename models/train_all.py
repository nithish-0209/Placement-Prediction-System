"""
Master Training Pipeline Orchestrator
Executes the full ML lifecycle across M1 through M6:
- Ingestion & 3-way Split
- Unified Preprocessing & Feature Engineering
- Linear vs Tree-based model training
- Unsupervised clustering & anomaly detection
- Probability Calibration & McNemar's significance testing
- Salary Regressor training
- Serializing production artifacts to models/saved/
"""

import os
import sys
import time
import json
import joblib

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.data_pipeline import load_raw_data, split_dataset, analyze_missing_values
from models.feature_engineering import build_preprocessing_pipeline, get_feature_names
from models.linear_models import (
    train_ridge_logistic, train_lasso_logistic, train_elastic_net_logistic,
    interpret_coefficients, compare_linear_regression_methods
)
from models.tree_models import (
    train_decision_tree, compute_ccp_path, train_random_forest,
    train_gradient_boosting, extract_tree_feature_importance, WHY_TREES_BEAT_NN_ON_TABULAR
)
from models.unsupervised import (
    compute_kmeans_elbow_silhouette, fit_dbscan, fit_pca,
    fit_isolation_forest, UNSUPERVISED_CONCEPTS
)
from models.evaluation import (
    evaluate_classification_metrics, evaluate_regression_metrics,
    compute_calibration_diagnostics, calibrate_model, run_mcnemar_test,
    compute_learning_curves
)
from models.monitoring import DataDriftDetector

SAVED_DIR = os.path.join(os.path.dirname(__file__), 'saved')

def run_master_pipeline():
    start_total_time = time.time()
    os.makedirs(SAVED_DIR, exist_ok=True)
    print("="*70)
    print("STARTING PLACEMENTPREDICT MASTER ML LIFECYCLE TRAINING PIPELINE")
    print("="*70)

    # 1. Load Data
    raw_df = load_raw_data()
    print(f"[Stage 1: Ingestion] Loaded raw dataset with {len(raw_df)} records.")
    missing_report = analyze_missing_values(raw_df)

    # 2. Three-way Split (M5)
    splits = split_dataset(raw_df, test_size=0.15, val_size=0.15, random_state=42)
    X_train_raw, y_train = splits['X_train'], splits['y_train']
    X_val_raw, y_val = splits['X_val'], splits['y_val']
    X_test_raw, y_test = splits['X_test'], splits['y_test']
    print(f"[Stage 2: Split] Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    # 3. Feature Engineering Pipeline (M2, M4, M6)
    print("[Stage 3: Feature Engineering] Fitting ColumnTransformer + KMeans Feature Adder...")
    prep_pipeline = build_preprocessing_pipeline(scaler_type='standard', add_cluster_features=True, n_clusters=4)
    X_train_trans = prep_pipeline.fit_transform(X_train_raw)
    X_val_trans = prep_pipeline.transform(X_val_raw)
    X_test_trans = prep_pipeline.transform(X_test_raw)
    feature_names = get_feature_names(prep_pipeline, X_train_raw)
    print(f"Engineered {X_train_trans.shape[1]} total features.")

    # 4. Linear Models (M2)
    print("[Stage 4: Linear Models] Training Ridge, Lasso, and ElasticNet...")
    ridge_clf = train_ridge_logistic(X_train_trans, y_train, C=1.0)
    lasso_clf = train_lasso_logistic(X_train_trans[:15000], y_train[:15000], C=0.3)
    elastic_clf = train_elastic_net_logistic(X_train_trans[:15000], y_train[:15000], C=0.5, l1_ratio=0.5)

    coef_interpretation = interpret_coefficients(ridge_clf, feature_names)
    lasso_sparse_summary = {
        'total_features': len(feature_names),
        'zeroed_out_by_lasso': int(np.sum(np.isclose(lasso_clf.coef_[0], 0.0, atol=1e-3)))
    }

    # 5. Tree Models (M3)
    print("[Stage 5: Tree Models] Training Decision Tree, Random Forest, HistGradientBoosting...")
    tree_clf = train_decision_tree(X_train_trans, y_train, criterion='gini', max_depth=6)
    ccp_path_data = compute_ccp_path(X_train_trans, y_train)
    rf_clf = train_random_forest(X_train_trans[:20000], y_train[:20000], n_estimators=80, max_depth=10)
    gb_clf = train_gradient_boosting(X_train_trans, y_train, learning_rate=0.08, max_iter=120)

    tree_feature_importance = extract_tree_feature_importance(gb_clf, feature_names, X_val_trans, y_val)

    # 6. Unsupervised Learning (M4)
    print("[Stage 6: Unsupervised Learning] Running KMeans, DBSCAN, PCA, and Isolation Forest...")
    kmeans_elbow_sil = compute_kmeans_elbow_silhouette(X_train_trans, k_range=range(2, 7))
    dbscan_report = fit_dbscan(X_train_trans, eps=3.0, min_samples=20)
    pca_report = fit_pca(X_train_trans, n_components=6)
    iso_forest = fit_isolation_forest(X_train_trans, contamination=0.02)

    # 7. Model Evaluation & Comparison (M5)
    print("[Stage 7: Model Evaluation] Benchmarking models on held-out test set...")
    models_to_test = {
        'Ridge Logistic': (ridge_clf, ridge_clf.predict(X_test_trans), ridge_clf.predict_proba(X_test_trans)[:, 1]),
        'Lasso Logistic': (lasso_clf, lasso_clf.predict(X_test_trans), lasso_clf.predict_proba(X_test_trans)[:, 1]),
        'Decision Tree (Pruned)': (tree_clf, tree_clf.predict(X_test_trans), tree_clf.predict_proba(X_test_trans)[:, 1]),
        'Random Forest': (rf_clf, rf_clf.predict(X_test_trans), rf_clf.predict_proba(X_test_trans)[:, 1]),
        'Gradient Boosting': (gb_clf, gb_clf.predict(X_test_trans), gb_clf.predict_proba(X_test_trans)[:, 1])
    }

    model_benchmarks = {}
    for name, (mdl, y_pred, y_prob) in models_to_test.items():
        metrics = evaluate_classification_metrics(y_test, y_pred, y_prob)
        model_benchmarks[name] = metrics

    # Statistical significance: McNemar test between Ridge and Gradient Boosting
    mcnemar_results = run_mcnemar_test(
        y_test,
        models_to_test['Ridge Logistic'][1],
        models_to_test['Gradient Boosting'][1]
    )

    # Probability Calibration (M5)
    print("[Stage 8: Calibration] Fitting Platt Scaling on best Gradient Boosting model...")
    uncalibrated_probs = models_to_test['Gradient Boosting'][2]
    uncalibrated_diag = compute_calibration_diagnostics(y_test, uncalibrated_probs)

    calibrated_gb = calibrate_model(gb_clf, X_val_trans, y_val, method='sigmoid')
    calibrated_probs = calibrated_gb.predict_proba(X_test_trans)[:, 1]
    calibrated_diag = compute_calibration_diagnostics(y_test, calibrated_probs)

    # Learning Curves (M5)
    learning_curves_data = compute_learning_curves(gb_clf, X_train_trans, y_train, cv=3)

    # Salary Package Regressor (M2 & M5)
    print("[Stage 9: Salary Regressor] Training regressor on placed cohort...")
    placed_mask_train = raw_df.loc[X_train_raw.index, 'placement_status'] == 1
    placed_mask_test = raw_df.loc[X_test_raw.index, 'placement_status'] == 1

    y_reg_train = raw_df.loc[X_train_raw.index[placed_mask_train], 'salary_package_lpa']
    X_reg_train_trans = X_train_trans[placed_mask_train]

    y_reg_test = raw_df.loc[X_test_raw.index[placed_mask_test], 'salary_package_lpa']
    X_reg_test_trans = X_test_trans[placed_mask_test]

    salary_regressor = Ridge(alpha=1.0)
    salary_regressor.fit(X_reg_train_trans, y_reg_train)
    reg_preds = salary_regressor.predict(X_reg_test_trans)
    salary_metrics = evaluate_regression_metrics(y_reg_test, reg_preds)
    
    linear_reg_comparison = compare_linear_regression_methods(
        X_reg_train_trans[:5000], y_reg_train[:5000],
        X_reg_test_trans[:1000], y_reg_test[:1000]
    )

    # 8. M6 Production Drift Baseline Setup
    print("[Stage 10: Production Baseline] Fitting DataDriftDetector on training baseline...")
    drift_detector = DataDriftDetector(X_train_raw)

    # 9. Packaging & Saving Artifacts (M6)
    print("[Stage 11: Packaging Artifacts] Saving pipelines and models to models/saved/...")
    joblib.dump(prep_pipeline, os.path.join(SAVED_DIR, 'pipeline.joblib'))
    joblib.dump(calibrated_gb, os.path.join(SAVED_DIR, 'best_model.joblib'))
    joblib.dump(ridge_clf, os.path.join(SAVED_DIR, 'linear_model.joblib'))
    joblib.dump(gb_clf, os.path.join(SAVED_DIR, 'tree_model.joblib'))
    joblib.dump(salary_regressor, os.path.join(SAVED_DIR, 'salary_regressor.joblib'))
    joblib.dump(iso_forest, os.path.join(SAVED_DIR, 'isolation_forest.joblib'))
    joblib.dump(pca_report['pca_model'], os.path.join(SAVED_DIR, 'pca_model.joblib'))
    joblib.dump(drift_detector, os.path.join(SAVED_DIR, 'drift_detector.joblib'))

    # Prepare summary JSON for Web UI rendering
    metrics_summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_dataset_rows': len(raw_df),
        'missing_values_report': missing_report,
        'model_benchmarks': model_benchmarks,
        'mcnemar_test': mcnemar_results,
        'calibration': {
            'uncalibrated': uncalibrated_diag,
            'calibrated': calibrated_diag
        },
        'salary_regression_metrics': salary_metrics,
        'linear_regression_comparison': linear_reg_comparison,
        'lasso_sparsity': lasso_sparse_summary,
        'top_coefficients': coef_interpretation.head(10).to_dict(orient='records'),
        'tree_feature_importance': {
            'features': tree_feature_importance['features'][:10],
            'permutation_importance': tree_feature_importance.get('permutation_importance', [])[:10]
        },
        'unsupervised': {
            'kmeans_elbow': kmeans_elbow_sil,
            'dbscan': dbscan_report,
            'pca': {
                'explained_variance': pca_report['explained_variance_ratio'],
                'cumulative_variance': pca_report['cumulative_variance_ratio']
            },
            'concepts': UNSUPERVISED_CONCEPTS
        },
        'ccp_path': ccp_path_data,
        'learning_curves': learning_curves_data,
        'why_trees_beat_nn': WHY_TREES_BEAT_NN_ON_TABULAR
    }

    with open(os.path.join(SAVED_DIR, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    # Model metadata contract (M6)
    model_metadata = {
        'model_name': 'PlacementPredict-HistGradientBoosting-Calibrated',
        'version': '1.0.0',
        'framework': 'scikit-learn 1.9.1',
        'pipeline_steps': ['MedianImputer+MissingIndicator', 'StandardScaler', 'OneHotEncoder', 'ClusterFeatureAdder', 'PlattCalibratedClassifier'],
        'input_features': list(X_train_raw.columns),
        'metrics': model_benchmarks['Gradient Boosting'],
        'training_duration_seconds': round(time.time() - start_total_time, 2)
    }

    with open(os.path.join(SAVED_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)

    print(f"Training successfully completed in {model_metadata['training_duration_seconds']}s!")
    print(f"Artifacts saved cleanly to: {SAVED_DIR}")

if __name__ == '__main__':
    run_master_pipeline()
