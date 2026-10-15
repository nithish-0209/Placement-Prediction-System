/**
 * PlacementPredict — Interactive Dashboard & Inference Controller
 * Handles Chart.js visualizations, Tab Switching, Drift Simulation,
 * Telemetry Updates, and Preset Fillers.
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabNavigation();
    initCharts();
    initMonitoringTriggers();
    initTelemetryPolling();
    initBatchPrediction();
});

// -------------------------------------------------------------
// Tab Switching
// -------------------------------------------------------------
function initTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
}

// -------------------------------------------------------------
// Chart.js Visualizations
// -------------------------------------------------------------
function initCharts() {
    const metricsDataEl = document.getElementById('metricsData');
    if (!metricsDataEl) return;

    let metrics = {};
    try {
        metrics = JSON.parse(metricsDataEl.textContent);
    } catch (e) {
        console.error('Failed to parse metrics data', e);
        return;
    }

    // Default Chart.js styling
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

    // 1. Target Class Balance Chart
    const classCtx = document.getElementById('classBalanceChart');
    if (classCtx) {
        new Chart(classCtx, {
            type: 'doughnut',
            data: {
                labels: ['Not Placed (Class 0)', 'Placed (Class 1)'],
                datasets: [{
                    data: [74.24, 25.76],
                    backgroundColor: ['#f43f5e', '#10b981'],
                    borderColor: '#141c2e',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 12, padding: 15 } }
                },
                cutout: '70%'
            }
        });
    }

    // 2. Tree Permutation Feature Importance Chart
    const treeCtx = document.getElementById('treeImportanceChart');
    if (treeCtx && metrics.tree_feature_importance) {
        const feats = metrics.tree_feature_importance.features || [];
        const imps = metrics.tree_feature_importance.permutation_importance || [];
        
        // Clean names for display
        const cleanNames = feats.map(f => f.replace('num__', '').replace('cat__', ''));

        new Chart(treeCtx, {
            type: 'bar',
            data: {
                labels: cleanNames,
                datasets: [{
                    label: 'Mean Decrease in F1',
                    data: imps,
                    backgroundColor: '#10b981',
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    y: { grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // 3. Calibration Reliability Diagram
    const calibCtx = document.getElementById('calibrationChart');
    if (calibCtx && metrics.calibration) {
        const uncalib = metrics.calibration.uncalibrated;
        const calib = metrics.calibration.calibrated;

        new Chart(calibCtx, {
            type: 'line',
            data: {
                labels: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                datasets: [
                    {
                        label: 'Perfect Calibration (y = x)',
                        data: [{x: 0, y: 0}, {x: 1, y: 1}],
                        borderColor: '#64748b',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Platt Calibrated',
                        data: calib.prob_pred.map((p, i) => ({x: p, y: calib.prob_true[i]})),
                        borderColor: '#10b981',
                        backgroundColor: '#10b981',
                        tension: 0.1,
                        pointRadius: 4
                    },
                    {
                        label: 'Raw Margin / Uncalibrated',
                        data: uncalib.prob_pred.map((p, i) => ({x: p, y: uncalib.prob_true[i]})),
                        borderColor: '#f59e0b',
                        backgroundColor: '#f59e0b',
                        tension: 0.1,
                        pointRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Mean Predicted Probability' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    y: { min: 0, max: 1, title: { display: true, text: 'Empirical Fraction of Positives' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                },
                plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } }
            }
        });
    }

    // 4. K-Means Elbow & Silhouette Chart
    const kmeansCtx = document.getElementById('kmeansChart');
    if (kmeansCtx && metrics.unsupervised && metrics.unsupervised.kmeans_elbow) {
        const km = metrics.unsupervised.kmeans_elbow;
        new Chart(kmeansCtx, {
            type: 'line',
            data: {
                labels: km.k_values.map(k => `K=${k}`),
                datasets: [
                    {
                        label: 'Silhouette Score (Right Axis)',
                        data: km.silhouette_scores,
                        borderColor: '#8b5cf6',
                        backgroundColor: '#8b5cf6',
                        yAxisID: 'y1',
                        pointRadius: 5
                    },
                    {
                        label: 'Inertia WCSS / 1000 (Left Axis)',
                        data: km.inertias.map(v => v / 1000),
                        borderColor: '#3b82f6',
                        backgroundColor: '#3b82f6',
                        yAxisID: 'y',
                        pointRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { type: 'linear', position: 'left', title: { display: true, text: 'Inertia (x1000)' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    y1: { type: 'linear', position: 'right', title: { display: true, text: 'Silhouette' }, grid: { drawOnChartArea: false } }
                },
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // 5. PCA Explained Variance Ratio Chart
    const pcaCtx = document.getElementById('pcaChart');
    if (pcaCtx && metrics.unsupervised && metrics.unsupervised.pca) {
        const pca = metrics.unsupervised.pca;
        new Chart(pcaCtx, {
            type: 'bar',
            data: {
                labels: pca.explained_variance.map((_, i) => `PC ${i+1}`),
                datasets: [
                    {
                        type: 'line',
                        label: 'Cumulative Variance Ratio',
                        data: pca.cumulative_variance,
                        borderColor: '#06b6d4',
                        backgroundColor: '#06b6d4',
                        pointRadius: 4,
                        yAxisID: 'y'
                    },
                    {
                        type: 'bar',
                        label: 'Individual Variance Ratio',
                        data: pca.explained_variance,
                        backgroundColor: 'rgba(59, 130, 246, 0.5)',
                        borderColor: '#3b82f6',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 1, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                },
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }
}

// -------------------------------------------------------------
// Production Drift & Retraining Triggers
// -------------------------------------------------------------
function initMonitoringTriggers() {
    const simDriftBtn = document.getElementById('simulateDriftBtn');
    const refreshDriftBtn = document.getElementById('refreshDriftBtn');
    const triggerRetrainBtn = document.getElementById('triggerRetrainBtn');

    if (simDriftBtn) {
        simDriftBtn.addEventListener('click', async () => {
            simDriftBtn.disabled = true;
            simDriftBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Injecting Drift Batch...';

            try {
                const res = await fetch('/api/simulate_drift', { method: 'POST' });
                const data = await res.json();
                updateDriftTable(data.drift_results);

                const kpiDrift = document.getElementById('kpiDrift');
                if (kpiDrift) {
                    kpiDrift.textContent = 'DRIFT DETECTED';
                    kpiDrift.className = 'kpi-value text-danger';
                }
                alert('Synthetic out-of-distribution drift injected! CGPA shifted downwards by 1.8 points. PSI > 0.2 threshold breached for key features.');
            } catch (err) {
                console.error('Error simulating drift', err);
            } finally {
                simDriftBtn.disabled = false;
                simDriftBtn.innerHTML = '<i class="fa-solid fa-biohazard"></i> Simulate Drift Batch';
            }
        });
    }

    if (refreshDriftBtn) {
        refreshDriftBtn.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/drift_report');
                const data = await res.json();
                updateDriftTable(data);
            } catch (err) {
                console.error('Error fetching drift report', err);
            }
        });
    }

    if (triggerRetrainBtn) {
        triggerRetrainBtn.addEventListener('click', async () => {
            if (!confirm('Trigger automated retraining pipeline across all M1-M6 steps?')) return;

            triggerRetrainBtn.disabled = true;
            triggerRetrainBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Retraining Pipeline...';

            try {
                const res = await fetch('/api/retrain', { method: 'POST' });
                const data = await res.json();
                alert(`Retraining complete in ${data.retraining_duration_seconds}s! Serving champion model updated.`);
                window.location.reload();
            } catch (err) {
                alert('Retraining failed: ' + err);
            } finally {
                triggerRetrainBtn.disabled = false;
                triggerRetrainBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Retrain Pipeline';
            }
        });
    }
}

function updateDriftTable(report) {
    const tbody = document.getElementById('driftTableBody');
    if (!tbody || !report || !report.features) return;

    tbody.innerHTML = '';
    const tsEl = document.getElementById('driftTimestamp');
    if (tsEl && report.timestamp) tsEl.textContent = `Evaluated at ${report.timestamp}`;

    for (const [col, stats] of Object.entries(report.features)) {
        const isDrift = stats.drift_status === 'HIGH_DRIFT';
        const isMod = stats.drift_status === 'MODERATE';
        const badgeClass = isDrift ? 'badge-danger' : (isMod ? 'badge-warning' : 'badge-success');

        const tr = document.createElement('tr');
        if (isDrift) tr.classList.add('highlight-row');
        tr.innerHTML = `
            <td><code>${col}</code></td>
            <td class="${isDrift ? 'text-danger font-bold' : ''}">${stats.psi}</td>
            <td>${stats.ks_statistic}</td>
            <td>${stats.ks_pvalue < 0.001 ? '< 0.001' : stats.ks_pvalue.toFixed(4)}</td>
            <td>${stats.baseline_mean}</td>
            <td>${stats.current_mean}</td>
            <td><span class="badge ${badgeClass}">${stats.drift_status}</span></td>
        `;
        tbody.appendChild(tr);
    }
}

// -------------------------------------------------------------
// Telemetry Live Polling
// -------------------------------------------------------------
function initTelemetryPolling() {
    setInterval(async () => {
        try {
            const res = await fetch('/api/telemetry');
            const data = await res.json();

            const kpiLat = document.getElementById('kpiLatency');
            if (kpiLat) kpiLat.textContent = `${data.avg_latency_ms} ms`;

            const monReq = document.getElementById('monTotalReq');
            if (monReq) monReq.textContent = data.total_requests;

            const monLat = document.getElementById('monAvgLat');
            if (monLat) monLat.textContent = `${data.avg_latency_ms} ms`;

            const monP95 = document.getElementById('monP95Lat');
            if (monP95) monP95.textContent = data.p95_latency_ms;

            const monThr = document.getElementById('monThroughput');
            if (monThr) monThr.textContent = `${data.throughput_req_per_sec} req/s`;

            const monRate = document.getElementById('monPlaceRate');
            if (monRate) monRate.textContent = `${data.placement_rate_predicted}%`;
        } catch (e) {
            // Ignore background polling errors
        }
    }, 6000);
}

// -------------------------------------------------------------
// Live Studio Presets
// -------------------------------------------------------------
function loadSamplePreset(presetType) {
    const presets = {
        elite: {
            cgpa: 9.25, iq: 132, coding_skills_score: 94.0, internships_count: 3,
            projects_count: 4, hackathons_attended: 3, soft_skills_score: 86.0,
            backlogs: 0, academic_percentage_10th: 92.5, academic_percentage_12th: 91.0,
            stream: 'Computer Science', college_tier: 'Tier 1', gender: 'Female',
            extracurricular_activities: 'Yes'
        },
        borderline: {
            cgpa: 6.85, iq: 98, coding_skills_score: 52.0, internships_count: 1,
            projects_count: 1, hackathons_attended: 0, soft_skills_score: 58.0,
            backlogs: 1, academic_percentage_10th: 71.0, academic_percentage_12th: 68.5,
            stream: 'Electronics & Comm', college_tier: 'Tier 2', gender: 'Male',
            extracurricular_activities: 'No'
        },
        anomaly: {
            cgpa: 9.85, iq: 84, coding_skills_score: 15.0, internships_count: 0,
            projects_count: 0, hackathons_attended: 0, soft_skills_score: 25.0,
            backlogs: 4, academic_percentage_10th: 55.0, academic_percentage_12th: 52.0,
            stream: 'Mechanical', college_tier: 'Tier 3', gender: 'Male',
            extracurricular_activities: 'No'
        }
    };

    const data = presets[presetType];
    if (!data) return;

    for (const [key, val] of Object.entries(data)) {
        const input = document.getElementById(key);
        if (input) {
            input.value = val;
        }
    }
}

// -------------------------------------------------------------
// Batch Prediction Simulation
// -------------------------------------------------------------
function initBatchPrediction() {
    const runBatchBtn = document.getElementById('runSampleBatchBtn');
    if (!runBatchBtn) return;

    runBatchBtn.addEventListener('click', async () => {
        runBatchBtn.disabled = true;
        runBatchBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing 100 Samples...';

        try {
            // Generate synthetic batch array
            const testBatch = [];
            for (let i = 0; i < 100; i++) {
                testBatch.push({
                    cgpa: 6.0 + Math.random() * 3.5,
                    iq: 85 + Math.floor(Math.random() * 50),
                    academic_percentage_10th: 60 + Math.random() * 35,
                    academic_percentage_12th: 60 + Math.random() * 35,
                    internships_count: Math.floor(Math.random() * 4),
                    projects_count: Math.floor(Math.random() * 5),
                    coding_skills_score: 20 + Math.random() * 75,
                    hackathons_attended: Math.floor(Math.random() * 3),
                    soft_skills_score: 40 + Math.random() * 55,
                    backlogs: Math.random() < 0.7 ? 0 : 1,
                    stream: ['Computer Science', 'Information Technology', 'Mechanical'][Math.floor(Math.random() * 3)],
                    college_tier: ['Tier 1', 'Tier 2', 'Tier 3'][Math.floor(Math.random() * 3)],
                    gender: 'Male',
                    extracurricular_activities: 'Yes'
                });
            }

            const res = await fetch('/api/batch_predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(testBatch)
            });
            const data = await res.json();

            document.getElementById('batchResultsContainer').style.display = 'block';
            document.getElementById('batchProcessed').textContent = data.total_processed;
            document.getElementById('batchLatency').textContent = `${data.batch_latency_ms} ms`;
            document.getElementById('batchPerSample').textContent = `${data.average_latency_per_sample_ms} ms`;
            document.getElementById('batchPlaced').textContent = `${data.placed_count} / ${data.total_processed} (${Math.round(data.placed_count/data.total_processed*100)}%)`;
        } catch (e) {
            alert('Batch prediction error: ' + e);
        } finally {
            runBatchBtn.disabled = false;
            runBatchBtn.innerHTML = '<i class="fa-solid fa-play"></i> Run 100-Sample Test Batch';
        }
    });
}
