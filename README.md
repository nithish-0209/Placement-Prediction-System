# PlacementPredict — End-to-End Classical ML & ML Engineering System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Container-Docker%20%2B%20Compose-2496ED.svg)](https://www.docker.com/)

A comprehensive, production-grade Machine Learning system designed to predict university student campus placement outcomes and estimated salary packages (LPA), serving as an exhaustive reference implementation covering the full ML System Lifecycle across modules **M1 through M6**.

---

## 📑 Curriculum & Syllabus Coverage

### M1. PlacementPredict as a System — The ML Lifecycle
- **Day-One Prediction Service**: A live prediction engine running immediately on application boot with sub-millisecond inference.
- **The ML System Lifecycle**: Tracing raw data $\to$ cleaned dataset $\to$ engineered features $\to$ training set $\to$ trained model $\to$ packaged artifact $\to$ served endpoint $\to$ monitored production $\to$ retraining loop.
- **Codebase Mapping**: Where each lifecycle stage lives:
  - Ingestion: `models/data_pipeline.py`
  - Feature Engineering: `models/feature_engineering.py`
  - Model Training: `models/train_all.py`
  - Serving: `app.py`
  - Monitoring: `models/monitoring.py`
- **Taxonomy**: Supervised classification ($y \in \{0, 1\}$) and regression ($y \in \mathbb{R}$) vs Unsupervised clustering vs Reinforcement Learning.
- **End-to-End Prediction Tracing**: HTTP Ingestion $\to$ Schema validation $\to$ Preprocessing $\to$ Platt-calibrated inference $\to$ Salary regression $\to$ Production telemetry.
- **Training-Serving Skew & Feature Stores**: Packaging a frozen, fitted `ColumnTransformer` (`pipeline.joblib`) to guarantee offline-online parity.
- **Downstream Boundaries**: Classical ML & ML Engineering (Here) vs Deep Learning (DLAC) vs Generative Models (GALE) vs LLMs (NLPL).

### M2. Supervised Learning — Linear Models at Depth
- **Linear Regression Comparison**: Closed-form Normal Equation $\theta = (X^T X)^{-1} X^T y$ vs Stochastic Gradient Descent (`SGDRegressor`) computational complexities ($O(n \cdot d^2 + d^3)$ vs $O(epochs \cdot n \cdot d)$).
- **Regularization**:
  - **Ridge ($L_2$)**: Shrinks weights smoothly towards zero, handles collinearity without dropping features.
  - **Lasso ($L_1$)**: Induces strict sparsity by driving uninformative coefficients to exactly 0 due to corner singularities of the diamond $L_1$ ball.
  - **ElasticNet**: Convex combination of $L_1$ and $L_2$ penalties.
- **Logistic Regression**: Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$, Binary Cross-Entropy loss, and hyperplane decision boundary.
- **Feature Scaling**: Detailed comparison of when `StandardScaler` matters (gradient descent, distance metrics) vs when it does not (decision trees).
- **Categorical & Missing Strategy**: `OneHotEncoder`, `OrdinalEncoder`, `SimpleImputer` alongside `MissingIndicator` binary feature generation.
- **Coefficient Interpretation**: Learned odds ratios $\exp(\beta)$ explaining feature impact on recruitment odds.

### M3. Supervised Learning — Tree-Based Models
- **Decision Trees**: Greedy top-down induction with Gini Impurity ($1 - \sum p_i^2$) and Entropy ($-\sum p_i \log_2 p_i$); pre-pruning via `max_depth` and post-pruning via Cost-Complexity Pruning (CCP $\alpha$).
- **Random Forests**: Bagging (bootstrap aggregation) + feature subsampling ($\sqrt{d}$) to reduce variance; Out-Of-Bag (OOB) generalization score.
- **Gradient Boosted Ensembles**: `HistGradientBoosting` (histogram-based binning into 256 bins, leaf-wise tree growth, shrinkage/learning rate).
- **Feature Importance**: Gini Impurity (MDI) vs Permutation Feature Importance evaluated on held-out validation data.
- **Why Trees Routinely Beat Neural Networks on Tabular Data**:
  1. Orthogonal step-function decision boundaries match discrete thresholds in structured tabular data.
  2. Invariance to monotonic feature transformations.
  3. Seamless handling of mixed numerical, ordinal, and categorical scales.
  4. Native handling of missing values without imputation distortions.

### M4. Unsupervised Learning
- **K-Means Clustering**: Lloyd's algorithm, K-Means++ initialization, Mini-Batch K-Means; Elbow method (Inertia/WCSS) and Silhouette score analysis across $K \in [2, 6]$.
- **Downstream Feature Injection**: Appending $K$-Means cluster distances and cluster labels as features into supervised models.
- **Hierarchical Clustering**: Agglomerative bottom-up clustering with Ward, Complete, Average, and Single linkage comparisons.
- **DBSCAN**: Density-based clustering with $\varepsilon$ and $minPts$; natural discovery of non-convex clusters and noise outlier identification.
- **Principal Component Analysis (PCA)**: Standardized orthogonal variance maximization; explained and cumulative variance ratios; highlighting the standardization pitfall.
- **t-SNE & UMAP Insights**: Why distances between clusters in t-SNE space are non-metric and cannot be interpreted as global distances.
- **Anomaly Detection**: `IsolationForest` flagging unusual candidate profiles (e.g. contradictory high academic scores with zero coding ability and multiple backlogs).

### M5. Model Evaluation, Selection, and Calibration
- **Three-Way Split Discipline**: 70% Training, 15% Validation, and 15% strictly Held-Out Test set.
- **Classification Metrics**: Highlighting why Accuracy lies under class imbalance (74% unplaced); tracking Precision, Recall, F1-Score, Specificity, ROC-AUC, and PR-AUC.
- **Regression Metrics**: RMSE, MAE, MAPE, and $R^2$ for salary package (LPA) prediction.
- **Probability Calibration**: Why a 0.7 prediction must mean 70% empirical probability; Platt Scaling (`CalibratedClassifierCV`) vs Isotonic Regression; Reliability diagrams and Brier score.
- **Statistical Significance**: McNemar's paired chi-squared test evaluating error discordance between Linear and Tree ensembles ($p < 0.05$).
- **Learning Curves**: Diagnosing underfitting vs overfitting across increasing sample subsets.

### M6. ML Engineering — From Notebook to Production
- **The Notebook-to-Production Gap**: Packaging code into modular python packages, configuration files, and REST services.
- **Training-Serving Skew Prevention**: Shared frozen pipeline artifact (`pipeline.joblib`).
- **Production Monitoring**: Real-time request buffer, latency percentiles (Mean & P95), and serving throughput.
- **Data Drift Detection**: Automated Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests alerting on feature distribution shifts ($PSI \ge 0.2$).
- **Retraining Strategies**: Scheduled retraining and drift-triggered automated retraining loops.
- **MLOps Governance**: Serialized `model_metadata.json` logging model version, schema, and performance benchmarks.

---

## 🏗️ Project Architecture

```
placement_prediction/
├── data/
│   ├── generate_dataset.py        # Generates 50,000 realistic student records
│   └── placement_data.csv         # 50,000-row verified dataset
├── models/
│   ├── __init__.py
│   ├── data_pipeline.py           # Ingestion, validation, 3-way stratified split
│   ├── feature_engineering.py     # ColumnTransformer, scalers, encoders, cluster features
│   ├── linear_models.py           # Ridge, Lasso (sparsity), ElasticNet, Normal Equations
│   ├── tree_models.py             # Decision Tree (CCP), Random Forest, HistGradientBoosting
│   ├── unsupervised.py            # KMeans, DBSCAN, PCA, Isolation Forest
│   ├── evaluation.py              # Metrics, Platt Calibration, McNemar Test, Learning Curves
│   ├── monitoring.py              # Telemetry, PSI & KS-Test Drift Detector
│   ├── train_all.py               # Master ML lifecycle orchestrator
│   └── saved/                     # Packaged production artifacts (.joblib & .json)
│       ├── best_model.joblib
│       ├── pipeline.joblib
│       ├── linear_model.joblib
│       ├── tree_model.joblib
│       ├── salary_regressor.joblib
│       ├── isolation_forest.joblib
│       ├── pca_model.joblib
│       ├── drift_detector.joblib
│       ├── metrics_summary.json
│       └── model_metadata.json
├── templates/
│   ├── base.html                  # Base layout, navbar, live status badges, footer
│   ├── dashboard.html             # Multi-block UI: EDA, Preprocessing, Models, Unsupervised, Monitoring
│   ├── predict.html               # Live prediction studio, execution tracer, batch inference
│   └── lifecycle.html             # Dedicated M1 ML lifecycle architecture blueprint
├── static/
│   ├── css/
│   │   └── style.css              # Dark-themed dashboard styling
│   └── js/
│       └── main.js                # Chart.js visualizations, drift simulation, live telemetry
├── app.py                         # Flask web app & REST API endpoints
├── Dockerfile                     # Production multi-worker container configuration
├── docker-compose.yml             # Docker compose deployment
├── requirements.txt               # Production dependencies
├── .dockerignore
└── .gitignore
```

---

## 🚀 Getting Started

### Option 1: Local Python Environment

1. **Activate Virtual Environment** (or create one):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Verify / Re-train Models (Optional)**:
   ```powershell
   python models/train_all.py
   ```

3. **Launch the Flask Application**:
   ```powershell
   python app.py
   ```

4. **Access the Web Dashboard**:
   Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

### Option 2: Docker Container Deployment

1. **Build and Run via Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Or build using Dockerfile**:
   ```bash
   docker build -t placement-predict:v1.0 .
   docker run -p 5000:5000 placement-predict:v1.0
   ```

3. **Verify Health**:
   ```bash
   curl http://localhost:5000/api/telemetry
   ```

---

## 📡 REST API Documentation

### 1. Online Single Prediction
- **Endpoint**: `POST /api/predict`
- **Content-Type**: `application/json`
- **Request Payload**:
  ```json
  {
    "cgpa": 8.5,
    "iq": 120,
    "coding_skills_score": 85.0,
    "internships_count": 2,
    "projects_count": 3,
    "hackathons_attended": 2,
    "soft_skills_score": 78.0,
    "backlogs": 0,
    "academic_percentage_10th": 88.0,
    "academic_percentage_12th": 85.0,
    "stream": "Computer Science",
    "college_tier": "Tier 1",
    "gender": "Male",
    "extracurricular_activities": "Yes"
  }
  ```
- **Response**:
  ```json
  {
    "placement_status": 1,
    "placement_label": "Placed",
    "calibrated_probability": 0.8842,
    "placement_percentage": "88.42%",
    "estimated_salary_package_lpa": 24.5,
    "anomaly_flag": false,
    "latency_ms": 1.45,
    "model_version": "1.0.0"
  }
  ```

### 2. High-Throughput Batch Prediction
- **Endpoint**: `POST /api/batch_predict`
- **Accepts**: JSON array of student objects or `multipart/form-data` CSV file.

### 3. Simulate Data Drift Batch
- **Endpoint**: `POST /api/simulate_drift`
- **Action**: Artificially shifts incoming distributions to demonstrate automated drift alarms.

### 4. Trigger Retraining Loop
- **Endpoint**: `POST /api/retrain`
- **Action**: Re-executes the complete training pipeline and refreshes live serving artifacts.

---

## 📊 Benchmark Summary (Held-Out Test Set: 7,500 Samples)

| Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Logistic ($L_2$)** | 79.51% | 66.36% | 41.46% | 0.5104 | 0.8131 | 0.6148 | 0.1428 |
| **Lasso Logistic ($L_1$)** | 79.49% | 65.79% | 42.49% | 0.5164 | 0.8127 | 0.6150 | 0.1429 |
| **Decision Tree (CCP Pruned)** | 75.44% | 54.49% | 28.26% | 0.3722 | 0.7188 | 0.4285 | 0.1682 |
| **Random Forest (Bagging)** | 80.20% | 68.42% | 43.12% | 0.5289 | 0.8194 | 0.6278 | 0.1384 |
| **HistGradientBoosting (Calibrated)** | **80.52%** | **68.85%** | **44.82%** | **0.5429** | **0.8228** | **0.6341** | **0.1352** |
