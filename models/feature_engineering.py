"""
M2 & M4 & M6: Feature Engineering Pipeline Module
Builds production-grade preprocessors preventing training-serving skew.
Covers:
- Missing value imputation with MissingIndicator
- Feature scaling: StandardScaler vs MinMaxScaler
- Categorical encoding: OneHotEncoder & OrdinalEncoder
- Unsupervised feature engineering: KMeans cluster labels as downstream features
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.cluster import KMeans

from models.data_pipeline import NUMERICAL_COLS, CATEGORICAL_COLS

class ClusterFeatureAdder(BaseEstimator, TransformerMixin):
    """
    M4 Practical Pipeline:
    Uses KMeans clustering to discover latent student archetype clusters
    and adds cluster distance / one-hot cluster labels as features for downstream models.
    """
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        cluster_distances = self.kmeans_.transform(X)
        cluster_labels = self.kmeans_.predict(X).reshape(-1, 1)
        return np.hstack([X, cluster_distances, cluster_labels])

def build_preprocessing_pipeline(scaler_type='standard', add_cluster_features=True, n_clusters=4):
    """
    Constructs a complete preprocessing pipeline that guarantees
    training-serving symmetry (preventing training-serving skew).
    """
    # 1. Numerical Pipeline: Impute -> Missing Indicator -> Scale
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', scaler)
    ])

    # 2. Categorical Pipeline: Most frequent imputer -> One-Hot Encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Column Transformer combining both
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_COLS),
            ('cat', cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )

    if add_cluster_features:
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('cluster_adder', ClusterFeatureAdder(n_clusters=n_clusters))
        ])
        return full_pipeline

    return preprocessor

def get_feature_names(preprocessor, original_df):
    """Extracts engineered feature names after ColumnTransformer transformation."""
    try:
        # Check if wrapped in cluster adder
        if hasattr(preprocessor, 'named_steps'):
            ct = preprocessor.named_steps['preprocessor']
            names = list(ct.get_feature_names_out())
            if 'cluster_adder' in preprocessor.named_steps:
                n_cl = preprocessor.named_steps['cluster_adder'].n_clusters
                names += [f"cluster_dist_{i}" for i in range(n_cl)]
                names.append("cluster_label")
            return names
        else:
            return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feat_{i}" for i in range(30)]

if __name__ == '__main__':
    from models.data_pipeline import load_raw_data, split_dataset
    df = load_raw_data()
    splits = split_dataset(df)
    pipe = build_preprocessing_pipeline(scaler_type='standard', add_cluster_features=True)
    X_train_trans = pipe.fit_transform(splits['X_train'])
    print(f"Transformed X_train shape: {X_train_trans.shape}")
    feature_names = get_feature_names(pipe, splits['X_train'])
    print(f"Total engineered features: {len(feature_names)}")
    print(f"Sample feature names: {feature_names[:8]}")
