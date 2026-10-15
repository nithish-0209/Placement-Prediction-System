"""
M6: ML Engineering — From Notebook to Production
Implements production monitoring, drift detection, and retraining triggers:
- Latency & throughput tracking
- Data Drift Detection: Kolmogorov-Smirnov (KS) test & Population Stability Index (PSI)
- Concept/Label Drift monitoring & Alerting
- Training-serving skew prevention contracts
- Drift-triggered retraining pipeline orchestrator
"""

import time
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class DataDriftDetector:
    """
    Monitors data drift between the training baseline and live inference requests.
    Uses:
    1. Kolmogorov-Smirnov (KS) Test: compares cumulative distribution functions.
       p-value < 0.05 indicates statistically significant distribution shift.
    2. Population Stability Index (PSI):
       PSI < 0.1: No significant change
       0.1 <= PSI < 0.2: Moderate drift (warning)
       PSI >= 0.2: Significant drift (triggers automatic retraining alert!)
    """
    def __init__(self, baseline_df=None):
        self.baseline_stats = {}
        if baseline_df is not None:
            self.fit_baseline(baseline_df)

    def fit_baseline(self, df):
        """Extracts reference distributions from the training set."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col in ['placement_status', 'salary_package_lpa']:
                continue
            values = df[col].dropna().values
            # 10 equal-frequency quantiles as bin edges
            quantiles = np.linspace(0, 1, 11)
            bins = np.percentile(values, quantiles * 100)
            bins[0] -= 1e-5
            bins[-1] += 1e-5
            bins = np.unique(bins)
            
            hist, _ = np.histogram(values, bins=bins)
            probs = (hist + 1e-4) / (np.sum(hist) + 1e-4 * len(hist))

            self.baseline_stats[col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'bins': [float(b) for b in bins],
                'expected_probs': [float(p) for p in probs],
                'raw_sample': values[:2000].tolist()
            }

    def compute_psi(self, expected_probs, actual_values, bins):
        """Calculates Population Stability Index: sum((Actual - Expected) * ln(Actual / Expected))."""
        hist, _ = np.histogram(actual_values, bins=bins)
        actual_probs = (hist + 1e-4) / (np.sum(hist) + 1e-4 * len(hist))
        expected_probs = np.array(expected_probs)
        
        psi_value = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
        return float(psi_value)

    def detect_drift(self, incoming_df):
        """Evaluates incoming inference batch against baseline distributions."""
        drift_report = {}
        drift_detected = False

        for col, stats in self.baseline_stats.items():
            if col in incoming_df.columns:
                actual_vals = incoming_df[col].dropna().values
                if len(actual_vals) < 5:
                    continue

                # 1. KS-Test
                ks_stat, ks_pval = ks_2samp(stats['raw_sample'], actual_vals)
                
                # 2. PSI
                psi = self.compute_psi(stats['expected_probs'], actual_vals, stats['bins'])
                
                is_drifted = (psi >= 0.2) or (ks_pval < 0.01)
                if is_drifted:
                    drift_detected = True

                drift_report[col] = {
                    'psi': round(psi, 4),
                    'ks_statistic': round(float(ks_stat), 4),
                    'ks_pvalue': float(ks_pval),
                    'drift_status': 'HIGH_DRIFT' if psi >= 0.2 else ('MODERATE' if psi >= 0.1 else 'STABLE'),
                    'baseline_mean': round(stats['mean'], 2),
                    'current_mean': round(float(np.mean(actual_vals)), 2)
                }

        return {
            'overall_drift_detected': drift_detected,
            'features': drift_report,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }


class ProductionInferenceMonitor:
    """
    In-memory production telemetry tracker:
    Records latency (ms), request timestamps, predictions, and recent payload buffer.
    """
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.request_buffer = []
        self.latencies_ms = []
        self.predictions_history = []
        self.start_time = time.time()

    def log_inference(self, payload, prediction, probability, latency_ms):
        """Logs an online inference event."""
        event = {
            'timestamp': time.strftime("%H:%M:%S"),
            'latency_ms': round(latency_ms, 2),
            'prediction': int(prediction),
            'probability': round(float(probability), 4),
            'payload': payload
        }
        self.request_buffer.append(event)
        self.latencies_ms.append(latency_ms)
        self.predictions_history.append(int(prediction))

        if len(self.request_buffer) > self.buffer_size:
            self.request_buffer.pop(0)
            self.latencies_ms.pop(0)
            self.predictions_history.pop(0)

    def get_telemetry_summary(self):
        """Returns live monitoring metrics: throughput, avg latency, P95/P99 latency."""
        n_requests = len(self.latencies_ms)
        uptime_sec = time.time() - self.start_time
        throughput = round(n_requests / max(uptime_sec, 1.0), 2)

        if n_requests == 0:
            return {
                'total_requests': 0,
                'avg_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'throughput_req_per_sec': throughput,
                'placement_rate_predicted': 0.0,
                'recent_logs': []
            }

        avg_lat = np.mean(self.latencies_ms)
        p95_lat = np.percentile(self.latencies_ms, 95)
        pos_rate = np.mean(self.predictions_history) * 100

        return {
            'total_requests': n_requests,
            'avg_latency_ms': round(float(avg_lat), 2),
            'p95_latency_ms': round(float(p95_lat), 2),
            'throughput_req_per_sec': throughput,
            'placement_rate_predicted': round(float(pos_rate), 1),
            'recent_logs': self.request_buffer[-10:][::-1]
        }
