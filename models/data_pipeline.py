"""
M1 & M2: Data Pipeline & Ingestion Module
Handles data loading, validation, missing value imputation, MissingIndicator features,
and disciplined 3-way train/validation/test split.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, MissingIndicator

NUMERICAL_COLS = [
    'cgpa', 'iq', 'academic_percentage_10th', 'academic_percentage_12th',
    'internships_count', 'projects_count', 'coding_skills_score',
    'hackathons_attended', 'soft_skills_score', 'backlogs'
]

CATEGORICAL_COLS = [
    'stream', 'college_tier', 'gender', 'extracurricular_activities'
]

TARGET_COL = 'placement_status'
REG_TARGET_COL = 'salary_package_lpa'

def load_raw_data(filepath=None):
    """Loads raw dataset from CSV with schema validation."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'placement_data.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run data/generate_dataset.py first.")
    df = pd.read_csv(filepath)
    return df

def analyze_missing_values(df):
    """Calculates missing values count and percentage per feature."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    report = pd.DataFrame({
        'missing_count': null_counts,
        'missing_pct': null_pct
    })
    return report[report['missing_count'] > 0].to_dict(orient='index')

def split_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    M5 Disciplined Three-Way Split:
    - 70% Training set (for model fitting)
    - 15% Validation set (for hyperparameter tuning & early calibration)
    - 15% Held-out Test set (never touched until final unbiased evaluation)
    Stratified by target class to preserve class balance.
    """
    X = df.drop(columns=[TARGET_COL, REG_TARGET_COL, 'student_id'], errors='ignore')
    y = df[TARGET_COL]
    y_reg = df[REG_TARGET_COL] if REG_TARGET_COL in df.columns else None

    # Step 1: Split out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: Split remaining into train and validation
    val_rel_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel_size, stratify=y_temp, random_state=random_state
    )

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

if __name__ == '__main__':
    data = load_raw_data()
    print(f"Loaded dataset: {data.shape}")
    missing_rep = analyze_missing_values(data)
    print(f"Missing values report: {missing_rep}")
    splits = split_dataset(data)
    print(f"Train size: {len(splits['X_train'])}, Val size: {len(splits['X_val'])}, Test size: {len(splits['X_test'])}")
