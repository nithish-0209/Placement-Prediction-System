"""
PlacementPredict — Flask Web Application & REST API
M1-M6 ML System Implementation:
- Serves interactive dashboard with separate blocks: EDA, Preprocessing, Models, Unsupervised, Monitoring
- Provides live inference endpoint with end-to-end tracing and probability calibration
- Exposes telemetry, data drift detection (PSI / KS-test), and automated retraining triggers
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Ensure models can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.monitoring import ProductionInferenceMonitor, DataDriftDetector
from models.train_all import run_master_pipeline, SAVED_DIR

app = Flask(__name__)
app.config['SECRET_KEY'] = 'placement-predict-ml-lifecycle-secret-key-2026'

# Global model artifacts
artifacts = {}
inference_monitor = ProductionInferenceMonitor(buffer_size=500)

def load_artifacts():
    """Loads packaged models, preprocessor pipelines, and training summaries."""
    global artifacts
    try:
        pipeline = joblib.load(os.path.join(SAVED_DIR, 'pipeline.joblib'))
        best_model = joblib.load(os.path.join(SAVED_DIR, 'best_model.joblib'))
        linear_model = joblib.load(os.path.join(SAVED_DIR, 'linear_model.joblib'))
        salary_regressor = joblib.load(os.path.join(SAVED_DIR, 'salary_regressor.joblib'))
        isolation_forest = joblib.load(os.path.join(SAVED_DIR, 'isolation_forest.joblib'))
        pca_model = joblib.load(os.path.join(SAVED_DIR, 'pca_model.joblib'))
        drift_detector = joblib.load(os.path.join(SAVED_DIR, 'drift_detector.joblib'))
        
        with open(os.path.join(SAVED_DIR, 'metrics_summary.json'), 'r') as f:
            metrics_summary = json.load(f)
            
        with open(os.path.join(SAVED_DIR, 'model_metadata.json'), 'r') as f:
            model_metadata = json.load(f)

        artifacts = {
            'pipeline': pipeline,
            'best_model': best_model,
            'linear_model': linear_model,
            'salary_regressor': salary_regressor,
            'isolation_forest': isolation_forest,
            'pca_model': pca_model,
            'drift_detector': drift_detector,
            'metrics_summary': metrics_summary,
            'model_metadata': model_metadata
        }
        print("Model artifacts and pipelines loaded successfully into memory.")
    except Exception as e:
        print(f"Artifact loading warning: {e}. Executing master training pipeline...")
        run_master_pipeline()
        load_artifacts()

# Initialize artifacts on start
load_artifacts()

# -------------------------------------------------------------
# Web Page Routes
# -------------------------------------------------------------

@app.route('/')
def dashboard():
    """Main UI Dashboard with separated blocks: Overview, EDA, Preprocessing, Models, Unsupervised, Monitoring."""
    metrics = artifacts.get('metrics_summary', {})
    metadata = artifacts.get('model_metadata', {})
    telemetry = inference_monitor.get_telemetry_summary()
    return render_template(
        'dashboard.html',
        metrics=metrics,
        metadata=metadata,
        telemetry=telemetry
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Interactive Live Prediction Studio with end-to-end tracing."""
    prediction_result = None
    form_data = None
    trace_steps = None

    if request.method == 'POST':
        start_time = time.time()
        try:
            # Parse form fields
            form_data = {
                'cgpa': float(request.form.get('cgpa', 7.5)),
                'iq': int(request.form.get('iq', 105)),
                'academic_percentage_10th': float(request.form.get('academic_percentage_10th', 75.0)),
                'academic_percentage_12th': float(request.form.get('academic_percentage_12th', 75.0)) if request.form.get('academic_percentage_12th') else np.nan,
                'internships_count': float(request.form.get('internships_count', 1)) if request.form.get('internships_count') else np.nan,
                'projects_count': int(request.form.get('projects_count', 2)),
                'coding_skills_score': float(request.form.get('coding_skills_score', 65.0)),
                'hackathons_attended': int(request.form.get('hackathons_attended', 1)),
                'soft_skills_score': float(request.form.get('soft_skills_score', 70.0)) if request.form.get('soft_skills_score') else np.nan,
                'backlogs': int(request.form.get('backlogs', 0)),
                'stream': request.form.get('stream', 'Computer Science'),
                'college_tier': request.form.get('college_tier', 'Tier 2'),
                'gender': request.form.get('gender', 'Male'),
                'extracurricular_activities': request.form.get('extracurricular_activities', 'Yes')
            }

            input_df = pd.DataFrame([form_data])

            # M1 Trace: Step-by-step lifecycle execution
            trace_steps = [
                {'step': '1. HTTP Request Received', 'detail': f"Ingested payload with 14 input attributes at {time.strftime('%H:%M:%S')}", 'status': 'success'},
                {'step': '2. Schema Validation & Null Check', 'detail': f"Validated types; handled {sum(pd.isna(v) for v in form_data.values())} missing fields via MissingIndicator strategy", 'status': 'success'},
            ]

            # Step 3: Feature Engineering Pipeline (impute -> scale -> encode -> cluster features)
            t_trans_start = time.time()
            X_trans = artifacts['pipeline'].transform(input_df)
            trans_duration = round((time.time() - t_trans_start) * 1000, 2)
            trace_steps.append({
                'step': '3. Preprocessing & Unsupervised Feature Lookup',
                'detail': f"Applied StandardScaler, OneHotEncoder, and KMeans Cluster assignment -> {X_trans.shape[1]} engineered features ({trans_duration}ms)",
                'status': 'success'
            })

            # Step 4: Calibrated Model Inference
            t_inf_start = time.time()
            prob = float(artifacts['best_model'].predict_proba(X_trans)[0][1])
            is_placed = int(prob >= 0.5)
            inf_duration = round((time.time() - t_inf_start) * 1000, 2)
            trace_steps.append({
                'step': '4. Calibrated Model Inference',
                'detail': f"HistGradientBoosting with Platt Sigmoid Calibration calculated P(Placed) = {prob*100:.2f}% ({inf_duration}ms)",
                'status': 'success'
            })

            # Step 5: Regression Salary package estimation (if placed or potential)
            pred_salary = None
            if is_placed:
                pred_salary = max(3.5, round(float(artifacts['salary_regressor'].predict(X_trans)[0]), 2))
                trace_steps.append({
                    'step': '5. Salary Regressor Inference',
                    'detail': f"Closed-form Ridge Regressor estimated package at {pred_salary} LPA",
                    'status': 'success'
                })
            else:
                pred_salary = 0.0

            # Step 6: Unsupervised Anomaly Detection check
            is_anomaly = artifacts['isolation_forest'].predict(X_trans)[0] == -1
            trace_steps.append({
                'step': '6. Isolation Forest Anomaly Check',
                'detail': "Outlier candidate profile detected!" if is_anomaly else "Normal profile within training manifold",
                'status': 'warning' if is_anomaly else 'success'
            })

            latency_ms = round((time.time() - start_time) * 1000, 2)
            trace_steps.append({
                'step': '7. Telemetry & Production Logging',
                'detail': f"Logged prediction event to telemetry buffer in {latency_ms}ms total latency",
                'status': 'success'
            })

            # Log to telemetry
            inference_monitor.log_inference(form_data, is_placed, prob, latency_ms)

            prediction_result = {
                'placement_status': is_placed,
                'probability': round(prob * 100, 2),
                'salary_package_lpa': pred_salary,
                'is_anomaly': is_anomaly,
                'latency_ms': latency_ms
            }

        except Exception as e:
            prediction_result = {'error': str(e)}

    return render_template(
        'predict.html',
        prediction=prediction_result,
        form_data=form_data,
        trace_steps=trace_steps
    )

@app.route('/lifecycle')
def lifecycle_page():
    """Dedicated ML System Lifecycle Architecture deep-dive (M1)."""
    return render_template('lifecycle.html')

# -------------------------------------------------------------
# REST API Endpoints (M1, M6)
# -------------------------------------------------------------

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for real-time online inference:
    Takes student features in JSON and returns calibrated prediction, probability,
    salary estimate, and latency.
    """
    start_time = time.time()
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({'error': 'No input payload provided'}), 400

    try:
        input_df = pd.DataFrame([payload])
        X_trans = artifacts['pipeline'].transform(input_df)
        prob = float(artifacts['best_model'].predict_proba(X_trans)[0][1])
        is_placed = int(prob >= 0.5)

        pred_salary = 0.0
        if is_placed:
            pred_salary = max(3.5, round(float(artifacts['salary_regressor'].predict(X_trans)[0]), 2))

        is_anomaly = bool(artifacts['isolation_forest'].predict(X_trans)[0] == -1)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        inference_monitor.log_inference(payload, is_placed, prob, latency_ms)

        return jsonify({
            'placement_status': is_placed,
            'placement_label': 'Placed' if is_placed else 'Not Placed',
            'calibrated_probability': round(prob, 4),
            'placement_percentage': f"{round(prob * 100, 2)}%",
            'estimated_salary_package_lpa': pred_salary,
            'anomaly_flag': is_anomaly,
            'latency_ms': latency_ms,
            'model_version': artifacts['model_metadata']['version']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """
    Batch prediction endpoint:
    Accepts CSV upload or JSON list of profiles, processes concurrently in batch mode.
    """
    start_time = time.time()
    df = None

    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
    elif request.is_json:
        payload = request.get_json()
        df = pd.DataFrame(payload)

    if df is None or len(df) == 0:
        return jsonify({'error': 'Valid CSV file or JSON array required'}), 400

    try:
        X_trans = artifacts['pipeline'].transform(df)
        probs = artifacts['best_model'].predict_proba(X_trans)[:, 1]
        preds = (probs >= 0.5).astype(int)
        salaries = artifacts['salary_regressor'].predict(X_trans)
        salaries = np.where(preds == 1, np.clip(np.round(salaries, 2), 3.5, 50.0), 0.0)

        results = []
        for i in range(min(len(df), 100)):  # return sample preview
            results.append({
                'index': i,
                'prediction': int(preds[i]),
                'probability': round(float(probs[i]), 4),
                'salary_lpa': float(salaries[i])
            })

        duration = round((time.time() - start_time) * 1000, 2)
        return jsonify({
            'total_processed': len(df),
            'batch_latency_ms': duration,
            'average_latency_per_sample_ms': round(duration / len(df), 3),
            'placed_count': int(np.sum(preds == 1)),
            'unplaced_count': int(np.sum(preds == 0)),
            'preview_samples': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/drift_report', methods=['GET'])
def api_drift_report():
    """Returns real-time data drift evaluation using incoming requests or test batch."""
    if len(inference_monitor.request_buffer) < 5:
        # Generate representative sample batch to demonstrate live drift calculation
        sample_df = pd.read_csv(os.path.join('data', 'placement_data.csv')).sample(200, random_state=42)
        report = artifacts['drift_detector'].detect_drift(sample_df)
    else:
        req_df = pd.DataFrame([e['payload'] for e in inference_monitor.request_buffer])
        report = artifacts['drift_detector'].detect_drift(req_df)

    return jsonify(report)

@app.route('/api/simulate_drift', methods=['POST'])
def api_simulate_drift():
    """
    Simulates production Out-Of-Distribution (OOD) data drift
    by artificially shifting CGPA and coding scores downward to demonstrate drift alarms.
    """
    raw_df = pd.read_csv(os.path.join('data', 'placement_data.csv')).sample(300, random_state=99)
    # Inject synthetic drift: shift CGPA downwards by 1.8 points and backlogs up
    raw_df['cgpa'] = np.clip(raw_df['cgpa'] - 1.8, 5.0, 10.0)
    raw_df['coding_skills_score'] = np.clip(raw_df['coding_skills_score'] - 25.0, 10.0, 99.0)
    raw_df['backlogs'] = raw_df['backlogs'] + 2

    drift_res = artifacts['drift_detector'].detect_drift(raw_df)
    return jsonify({
        'status': 'Simulated Drift Batch Injected',
        'sample_count': len(raw_df),
        'drift_results': drift_res
    })

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """Triggers the automated retraining loop (M6 Retraining strategy)."""
    t0 = time.time()
    run_master_pipeline()
    load_artifacts()
    duration = round(time.time() - t0, 2)
    return jsonify({
        'status': 'Retraining completed successfully',
        'retraining_duration_seconds': duration,
        'new_metrics': artifacts['model_metadata']['metrics'],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/telemetry', methods=['GET'])
def api_telemetry():
    """Returns live monitoring telemetry."""
    return jsonify(inference_monitor.get_telemetry_summary())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting PlacementPredict Flask Application on http://127.0.0.1:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)
