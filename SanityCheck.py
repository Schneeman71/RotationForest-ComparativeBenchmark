import pandas as pd
import numpy as np
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer, 
                              load_digits, load_diabetes)

# --- Configuration & Seed Generation ---
GLOBAL_SEED = 123
_rng = np.random.RandomState(GLOBAL_SEED)
# Generate 10 high-entropy seeds for the benchmark
SEEDS = _rng.randint(0, 1_000_000, size=10).tolist()

# --- Dataset Loaders (Exploring Different Attributes) ---

def load_iris_ds():
    """Classification: Low-dimensional, highly separable."""
    data = load_iris()
    return data.data, data.target, "classification", "Iris"

def load_wine_ds():
    """Classification: Chemical features, 3 classes."""
    data = load_wine()
    return data.data, data.target, "classification", "Wine"

def load_breast_cancer_ds():
    """Classification: Binary, clinical/diagnostic features."""
    data = load_breast_cancer()
    return data.data, data.target, "classification", "Breast Cancer"

def load_digits_ds():
    """Classification: High-dimensional (64 features), non-linear pixel patterns."""
    data = load_digits()
    return data.data, data.target, "classification", "Digits"

def load_diabetes_ds():
    """Regression: Continuous target, 10 physiological features."""
    data = load_diabetes()
    return data.data, data.target, "regression", "Diabetes"

def run_benchmarks():
    dataset_loaders = [
        load_iris_ds, 
        load_wine_ds, 
        load_breast_cancer_ds, 
        load_digits_ds, 
        load_diabetes_ds
    ]
    
    print(f"Starting Multi-Dataset Benchmark | Global Seed: {GLOBAL_SEED}")
    print(f"Iterative Seeds: {SEEDS}\n")

    for loader in dataset_loaders:
        X, y, task_type, ds_name = loader()
        
        # Scaling is essential for Rotation Forest's internal PCA
        X = StandardScaler().fit_transform(X)
        
        dataset_results = []
        print(f"--- Running Benchmark: {ds_name} ({task_type}) ---")

        for seed in SEEDS:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            
            for name in ["rf", "et", "rotf"]:
                t_start = time.time()
                
                # Lazy Imports for worker-style efficiency
                from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                                              ExtraTreesClassifier, ExtraTreesRegressor)
                from aeon.classification.sklearn import RotationForestClassifier
                from aeon.regression.sklearn import RotationForestRegressor
                
                # Model Factory based on Task Type
                if task_type == "classification":
                    if name == "rf": model = RandomForestClassifier(n_estimators=100, random_state=seed)
                    elif name == "et": model = ExtraTreesClassifier(n_estimators=100, random_state=seed)
                    elif name == "rotf": model = RotationForestClassifier(n_estimators=25, random_state=seed)
                    scoring_fn = accuracy_score
                    metric_name = "Accuracy"
                else:
                    if name == "rf": model = RandomForestRegressor(n_estimators=100, random_state=seed)
                    elif name == "et": model = ExtraTreesRegressor(n_estimators=100, random_state=seed)
                    elif name == "rotf": model = RotationForestRegressor(n_estimators=25, random_state=seed)
                    scoring_fn = lambda y_t, p: np.sqrt(mean_squared_error(y_t, p))
                    metric_name = "RMSE"

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = scoring_fn(y_test, preds)
                duration = time.time() - t_start
                
                dataset_results.append({
                    "Model": name.upper(),
                    "Score": score,
                    "Time": duration
                })
                
                del model
                gc.collect()

        # --- Aggregate and Print Results for the Dataset ---
        df_ds = pd.DataFrame(dataset_results)
        summary = df_ds.groupby("Model").agg({"Score": ["mean", "std"], "Time": "mean"})
        summary.columns = [f'Mean {metric_name}', 'Std Dev (Stability)', 'Avg Time (s)']
        
        # Sort: Best is High for Accuracy, Low for RMSE
        ascending = True if metric_name == "RMSE" else False
        summary = summary.sort_values(by=f'Mean {metric_name}', ascending=ascending)

        print(summary.to_string())
        print("-" * 70 + "\n")

if __name__ == "__main__":
    run_benchmarks()