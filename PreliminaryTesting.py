import openml
import pandas as pd
import numpy as np
import os
import traceback
import time
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor)

from aeon.classification.sklearn import RotationForestClassifier
from aeon.regression.sklearn import RotationForestRegressor
from aeon.datasets import (load_classification, load_regression, tsc_datasets, tser_datasets)

# --- Hyperparameters and Configuration ---

# Hardware / Processing
N_JOBS = -1               # -1 = use all cores
SEEDS = [0, 1, 2, 3, 4]   # Seeds to iterate over

# Dataset Loading
DATASET_LIMIT = -1         

# Model Hyperparameters
N_ESTIMATORS_RF = 100     # Random Forest Trees
N_ESTIMATORS_ET = 100     # Extra Trees (Number of trees)
N_ESTIMATORS_ROTF = 50    # Rotation Forest Trees 
ROTF_TIME_LIMIT = 20      # Max minutes RotF is allowed to build trees

# Preprocessing
ENCODER_TYPE = "onehot"   # Alternative = 'target' (for high cardinality data)


# --- Helper Functions for Preprocessing ---

def get_feature_type(X):
    """Determines if features are Continuous, Categorical, or Mixed."""
    if isinstance(X, np.ndarray):
        return "Continuous"
    
    # For DataFrames
    numerics = X.select_dtypes(include="number").shape[1]
    objects = X.select_dtypes(exclude="number").shape[1]
    
    if numerics > 0 and objects == 0:
        return "Continuous"
    elif numerics == 0 and objects > 0:
        return "Categorical"
    else:
        return "Mixed"

def get_imbalance_metrics(y, task):
    """Calculates class imbalance ratio for classification."""
    if task == "regression":
        return "N/A"
    
    try:
        # Convert to Series if numpy
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        counts = y.value_counts()
        if len(counts) == 0: return "N/A"
        
        ratio = counts.max() / counts.min()
        return round(ratio, 4)
    except Exception:
        return "Error"


# --- Dataset Loaders ---

def load_openml_suite(suite_id, task, suite_name, limit=2):
    suite = openml.study.get_suite(suite_id)
    datasets = []
    task_iterator = iter(suite.tasks)
    
    # Allow -1 to specify all datasets
    target_count = float('inf') if limit == -1 else limit
    
    while len(datasets) < target_count:
        try:
            try:
                task_id = next(task_iterator)
            except StopIteration:
                break 

            task_obj = openml.tasks.get_task(task_id)
            dataset = task_obj.get_dataset()

            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe"
            )

            if task == "regression" and not pd.api.types.is_numeric_dtype(y):
                continue

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

def load_aeon_tsc_suite(limit=5):
    names = iter(tsc_datasets.univariate_equal_length + tsc_datasets.multivariate_equal_length)
    datasets = []
    target_count = float('inf') if limit == -1 else limit

    while len(datasets) < target_count:
        try:
            name = next(names)
            X_train, y_train = load_classification(name, split="train")
            X_test, y_test = load_classification(name, split="test")

            datasets.append({
                "id": f"tsc_{name}",
                "name": name,
                "suite_name": "AEON-TSC",
                "task": "classification",
                "data_type": "ts", 
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
            })
        except StopIteration: break
        except Exception: continue
    return datasets

def load_aeon_tser_suite(limit=5):
    names = iter(list(tser_datasets.tser_monash.keys()) + list(tser_datasets.tser_soton_clean))
    datasets = []
    target_count = float('inf') if limit == -1 else limit

    while len(datasets) < target_count:
        try:
            name = next(names)
            X_train, y_train = load_regression(name, split="train")
            X_test, y_test = load_regression(name, split="test")

            datasets.append({
                "id": f"tser_{name}",
                "name": name,
                "suite_name": "AEON-TSER",
                "task": "regression",
                "data_type": "ts",
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
            })
        except StopIteration: break
        except Exception: continue
    return datasets


# --- Preprocessing Utilities ---

def flatten_time_series_fast(X):
    n, c, t = X.shape
    return X.reshape(n, c * t)

def ensure_numeric_format(X, data_type):
    if data_type == "tabular": return X
    if data_type == "ts":
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return flatten_time_series_fast(X)
        if isinstance(X, pd.DataFrame):
            try:
                channels = [np.stack(X[col].tolist()) for col in X.columns]
                X_3d = np.stack(channels, axis=1)
                return flatten_time_series_fast(X_3d)
            except ValueError: pass
    if isinstance(X, pd.DataFrame): return X.to_numpy()
    return X

def make_smart_preprocessor(X, task):
    if isinstance(X, np.ndarray):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    te_type = "continuous" if task == "regression" else "multiclass"
    numeric_features = X.select_dtypes(include="number").columns
    categorical_features = X.select_dtypes(exclude="number").columns

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    
    if ENCODER_TYPE == "target":
        encoder = TargetEncoder(target_type=te_type, random_state=42)
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

def make_model(model_name, task, preprocessor, seed):
    if model_name == "rf":
        est_class = RandomForestClassifier if task == "classification" else RandomForestRegressor
        model = est_class(n_estimators=N_ESTIMATORS_RF, n_jobs=1, random_state=seed)
    elif model_name == "et":
        est_class = ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor
        model = est_class(n_estimators=N_ESTIMATORS_ET, n_jobs=1, random_state=seed)
    elif model_name == "rotf":
        est_class = RotationForestClassifier if task == "classification" else RotationForestRegressor
        model = est_class(
            n_estimators=N_ESTIMATORS_ROTF, 
            time_limit_in_minutes=ROTF_TIME_LIMIT, 
            n_jobs=1, 
            random_state=seed
        )
    
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model) 
    ])


# --- Worker Function for Joblib ---

def process_benchmark_task(ds, model_name, seed):
    try:
        t_start = time.time()
        
        # 1. Prepare Data
        X_train_orig = ensure_numeric_format(ds["X_train"], ds["data_type"])
        X_test_orig = ensure_numeric_format(ds["X_test"], ds["data_type"])
        y_train, y_test = ds["y_train"], ds["y_test"]

        # 2. Handle NaNs in Target (Regression)
        if ds["task"] == "regression":
            if isinstance(y_train, pd.Series): mask = y_train.notna()
            else: mask = ~np.isnan(y_train)
            if not mask.all():
                X_train_orig = X_train_orig[mask]
                y_train = y_train[mask]

        # 3. Handle Label Encoding (Classification)
        if ds["task"] == "classification":
            le = LabelEncoder()
            le.fit(y_train)
            y_train = le.transform(y_train)
            try:
                y_test = le.transform(y_test)
            except ValueError:
                test_mask = np.isin(ds["y_test"], le.classes_)
                X_test_orig = X_test_orig[test_mask]
                y_test = le.transform(ds["y_test"][test_mask])

        # 4. Extract Metadata
        feature_type = get_feature_type(ds["X_train"]) 
        imbalance_ratio = get_imbalance_metrics(ds["y_train"], ds["task"])
        rows = X_train_orig.shape[0] + X_test_orig.shape[0]
        cols = X_train_orig.shape[1]

        # 5. Build & Run
        preprocessor = make_smart_preprocessor(X_train_orig, ds["task"])
        pipe = make_model(model_name, ds["task"], preprocessor, seed)
        
        pipe.fit(X_train_orig, y_train)
        preds = pipe.predict(X_test_orig)
        
        if ds["task"] == "classification":
            metric_val = accuracy_score(y_test, preds)
            metric_name = "Accuracy"
        else:
            metric_val = np.sqrt(mean_squared_error(y_test, preds))
            metric_name = "RMSE"

        duration = time.time() - t_start

        return {
            "tablename": ds["name"],
            "classification_name": ds["suite_name"],
            "feature_type": feature_type,
            "model_type": model_name,
            "metric_type": metric_name,
            "dataset_rows": rows,
            "dataset_columns": cols,
            "time_taken": round(duration, 4),
            "imbalance_ratio": imbalance_ratio,
            "random_state": seed,
            "score": round(metric_val, 6),
            "status": "success"
        }

    except Exception as e:
        return {"tablename": ds["name"], "model_type": model_name, "random_state": seed, "status": "failed", "error": str(e)}


# --- Primary Execution ---

if __name__ == "__main__":
    print("Loading datasets...")
    datasets = []
    
    # OpenML CC-18 (Classification)
    datasets += load_openml_suite(99, task="classification", suite_name="OpenML-CC18", limit=DATASET_LIMIT)
    # OpenML 297 (Regression)
    datasets += load_openml_suite(297, task="regression", suite_name="OpenML-297", limit=DATASET_LIMIT)
    # AEON TSC
    datasets += load_aeon_tsc_suite(limit=DATASET_LIMIT)
    # AEON TSER
    datasets += load_aeon_tser_suite(limit=DATASET_LIMIT)
    
    print(f"Total Datasets Loaded: {len(datasets)}")
    print(f"Configurations: {len(datasets)} Datasets x 3 Models x {len(SEEDS)} Seeds = {len(datasets)*3*len(SEEDS)} Tasks")
    print(f"Using Encoder: {ENCODER_TYPE}")
    print("Starting Parallel Benchmarking...")

    # Create comprehensive task list
    tasks = [
        (ds, model, seed)
        for ds in datasets
        for model in ["rf", "et", "rotf"]
        for seed in SEEDS
    ]

    # Run Parallel
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_benchmark_task)(ds, model, seed) for ds, model, seed in tasks
    )

    # Save Results
    df = pd.DataFrame(results)
    if not df.empty:
        os.makedirs("results", exist_ok=True)
        filename = "results/benchmark_official_v1.csv"
        df.to_csv(filename, index=False)
        print(f"\nProcessing Complete. Saved {len(df)} rows to {filename}")

        print(df[["tablename", "model_type", "random_state", "score", "time_taken"]].head())