import pandas as pd
import numpy as np
import os
import time
import gc

N_JOBS = -1               

GLOBAL_SEED = 42  
_rng = np.random.RandomState(GLOBAL_SEED)

SEEDS = _rng.randint(0, 1_000_000, size=5).tolist()

DATASET_LIMIT = -1         
MAX_CELLS_PER_DATASET = 5_000_000  
ENCODER_TYPE = "onehot"   

print(f"Global Seed: {GLOBAL_SEED} | Iterative Seeds: {SEEDS}")

def get_openml():
    import openml
    return openml

def get_aeon_loaders(task_type="classification"):
    from aeon.datasets import load_classification, load_regression
    return load_classification if task_type == "classification" else load_regression

# --- Optimized Preprocessing Utilities ---

def flatten_time_series_fast(X):
    """Uses a memory-efficient view rather than a copy."""
    if isinstance(X, np.ndarray) and X.ndim == 3:
        n, c, t = X.shape
        return X.reshape(n, c * t)
    return X

def ensure_numeric_format(X, data_type):
    """Minimizes data duplication in RAM during conversion."""
    if data_type == "tabular": 
        return X
    
    if data_type == "ts":
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return flatten_time_series_fast(X)
        
        if isinstance(X, pd.DataFrame):
            try:
                # Convert nested DF to 3D Numpy then flatten
                X_3d = np.stack([np.stack(X[col].values) for col in X.columns], axis=1)
                return flatten_time_series_fast(X_3d)
            except (ValueError, TypeError):
                pass
                
    return X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)

# --- Metadata-Aware Loaders ---

def load_openml_suite(suite_id, task, suite_name, limit=-1):
    oml = get_openml()
    suite = oml.study.get_suite(suite_id)
    datasets = []
    
    target_count = float('inf') if limit == -1 else limit
    
    for task_id in suite.tasks:
        if len(datasets) >= target_count:
            break
            
        try:
            task_obj = oml.tasks.get_task(task_id)
            dataset = task_obj.get_dataset()
            
            # Check qualities before downloading data to save time/bandwidth
            q = dataset.qualities
            est_cells = int(q.get('NumberOfInstances', 0)) * int(q.get('NumberOfFeatures', 0))
            
            if est_cells > MAX_CELLS_PER_DATASET:
                print(f"  -> Skipping {dataset.name} ({est_cells:,} cells > limit)")
                continue

            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe"
            )

            train_idx, test_idx = task_obj.get_train_test_split_indices(fold=0)

            datasets.append({
                "id": f"openml_{dataset.dataset_id}",
                "name": dataset.name,
                "suite_name": suite_name,
                "task": task,
                "data_type": "tabular", 
                "X_train": X.iloc[train_idx],
                "X_test": X.iloc[test_idx],
                "y_train": y.iloc[train_idx],
                "y_test": y.iloc[test_idx],
            })
            print(f"Loaded {suite_name}: {dataset.name}")
            
        except Exception:
            continue
            
    return datasets

def load_aeon_suite(suite_type="TSC", limit=-1):
    from aeon.datasets import tsc_datasets, tser_datasets
    loader = get_aeon_loaders("classification" if suite_type == "TSC" else "regression")
    
    if suite_type == "TSC":
        names = tsc_datasets.univariate_equal_length + tsc_datasets.multivariate_equal_length
        task = "classification"
    else:
        names = list(tser_datasets.tser_monash.keys()) + list(tser_datasets.tser_soton_clean)
        task = "regression"

    datasets = []
    target_count = float('inf') if limit == -1 else limit

    for name in names:
        if len(datasets) >= target_count:
            break
        try:
            X_train, y_train = loader(name, split="train")
            
            if X_train.size > MAX_CELLS_PER_DATASET:
                print(f"  -> Skipping {name} ({X_train.size:,} cells > limit)")
                continue
                
            X_test, y_test = loader(name, split="test")

            datasets.append({
                "id": f"{suite_type.lower()}_{name}",
                "name": name,
                "suite_name": f"AEON-{suite_type}",
                "task": task,
                "data_type": "ts", 
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
            })
            print(f"Loaded {suite_type}: {name}")
        except Exception:
            continue
    return datasets

# --- Model & Pipeline Factory ---

def make_model(model_name, task, preprocessor, seed):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
    from aeon.classification.sklearn import RotationForestClassifier
    from aeon.regression.sklearn import RotationForestRegressor

    # Models use the iterative 'seed' passed from the SEEDS list
    if model_name == "rf":
        est = RandomForestClassifier if task == "classification" else RandomForestRegressor
        model = est(n_estimators=50, n_jobs=1, random_state=seed)
    elif model_name == "et":
        est = ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor
        model = est(n_estimators=50, n_jobs=1, random_state=seed)
    elif model_name == "rotf":
        est = RotationForestClassifier if task == "classification" else RotationForestRegressor
        model = est(n_estimators=25, n_jobs=1, random_state=seed)

    
    from sklearn.pipeline import Pipeline
    return Pipeline([("preprocess", preprocessor), ("model", model)])

# --- Optimized Preprocessor ---

def make_smart_preprocessor(X, task):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder

    if isinstance(X, np.ndarray):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())
        ])

    numeric_features = X.select_dtypes(include="number").columns
    categorical_features = X.select_dtypes(exclude="number").columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler())
    ])
    
    if ENCODER_TYPE == "target":
        encoder = TargetEncoder(target_type="continuous" if task == "regression" else "multiclass")
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])