import os
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Import utilities from your existing testing file
from PreliminaryTesting import (
    load_openml_suite, load_aeon_tsc_suite, load_aeon_tser_suite,
    ensure_numeric_format, make_smart_preprocessor, make_model,
    SEEDS, DATASET_LIMIT, N_JOBS
)

# --- Configuration ---
MAX_CELLS = 5_000_000  # Safety Cap: 5 million data points (Rows * Cols)

# --- Hyperparameter Search Spaces ---

def get_param_grid(model_name):
    """
    Defines the search space for tuning.
    Optimized to test specific hypotheses about Data Structure, Noise, and Redundancy.
    """
    if model_name in ["rf", "et"]:
        return {
            # Tests Feature Visibility: "sqrt" (Standard) vs 0.5 (Aggressive) vs 1.0 (Bagging/Regression)
            "model__max_features": ["sqrt", 0.5, 1.0],
            # Tests Smoothing: 1 (High Variance) vs 5 (Smoothed/Noisy Data)
            "model__min_samples_leaf": [1, 5]
        }
    elif model_name == "rotf":
        return {
            # 20 vs 50: Checks if added computation yields diminishing returns
            "model__n_estimators": [20, 50],
            # 3 vs 7: Checks Local (3) vs Global (7) structure preference
            "model__max_group": [3, 7],
            # 0.25 -> 0.75: Checks Robustness to Redundancy
            "model__remove_proportion": [0.25, 0.5, 0.75]
        }
    return {}

# --- Worker Function ---

def process_benchmark_grid(ds, model_name, seed):
    """Runs GridSearchCV and extracts the score AND runtime for EVERY parameter combination."""
    try:
        t_start = time.time()
        
        X_train_orig = ensure_numeric_format(ds["X_train"], ds["data_type"])
        y_train = ds["y_train"]

        # 1. Handle NaNs in Target (Regression)
        if ds["task"] == "regression":
            if isinstance(y_train, pd.Series): mask = y_train.notna()
            else: mask = ~np.isnan(y_train)
            if not mask.all():
                X_train_orig = X_train_orig[mask]
                y_train = y_train[mask]

        # 2. Basic Label Encoding for Classification
        if ds["task"] == "classification":
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)

        # 3. Build Tuned Model Pipeline
        preprocessor = make_smart_preprocessor(X_train_orig, ds["task"])
        base_pipe = make_model(model_name, ds["task"], preprocessor, seed)
        
        # 4. CV Strategy Setup
        n_folds = 3
        if ds["task"] == "classification":
            min_class_count = pd.Series(y_train).value_counts().min()
            if min_class_count < n_folds:
                cv_strategy = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            else:
                cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            scoring = "accuracy"
        else:
            cv_strategy = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            scoring = "neg_root_mean_squared_error"

        # 5. GridSearch Execution
        search = GridSearchCV(
            estimator=base_pipe,
            param_grid=get_param_grid(model_name),
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=1 
        )
        search.fit(X_train_orig, y_train)

        # 6. Extract all results
        cv_results = search.cv_results_
        total_grid_time = time.time() - t_start
        
        record = {
            "dataset": ds["name"],
            "seed": seed,
            "time_taken": round(total_grid_time, 4),
            "status": "success"
        }
        
        # Map parameter strings to their scores AND execution times
        for i, params in enumerate(cv_results["params"]):
            param_str = str(params).replace("model__", "") 
            
            # Score logic
            score = cv_results["mean_test_score"][i]
            record[param_str] = abs(score) if ds["task"] == "regression" else score
            
            # Time logic: mean fit time + mean score time (per fold), multiplied by folds
            time_per_fold = cv_results["mean_fit_time"][i] + cv_results["mean_score_time"][i]
            record[f"time_{param_str}"] = round(time_per_fold * n_folds, 4)
            
        return record

    except Exception as e:
        return {"dataset": ds.get("name"), "seed": seed, "status": "failed", "error": str(e)}

# --- Execution & Aggregation ---

if __name__ == "__main__":
    # === CONFIGURATION BLOCK ===
    SELECTED_MODEL = "rotf"  # "rf", "et", or "rotf"
    BENCHMARKS = ["OpenML-CC18", "OpenML-297", "AEON-TSC", "AEON-TSER"]
    # ===========================
    
    # Generate a master timestamp for this entire run batch
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    
    print(f"Starting Multi-Benchmark GridSearch for Model: {SELECTED_MODEL.upper()}")
    
    for benchmark in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"   PROCESSING BENCHMARK: {benchmark}")
        print(f"{'='*60}")
        
        # 1. Load Datasets
        datasets = []
        if benchmark == "OpenML-CC18":
            datasets += load_openml_suite(99, "classification", "OpenML-CC18", limit=DATASET_LIMIT)
        elif benchmark == "OpenML-297":
            datasets += load_openml_suite(297, "regression", "OpenML-297", limit=DATASET_LIMIT)
        elif benchmark == "AEON-TSC":
            datasets += load_aeon_tsc_suite(limit=DATASET_LIMIT)
        elif benchmark == "AEON-TSER":
            datasets += load_aeon_tser_suite(limit=DATASET_LIMIT)

        if not datasets:
            print(f"Warning: No datasets loaded for {benchmark}. Skipping...")
            continue
            
        # 2. Filter Datasets (Complexity Guardrail)
        valid_datasets = []
        for ds in datasets:
            # Quick shape check (approximate for TS data if not flat yet)
            n_rows = len(ds["X_train"])
            try:
                n_cols = ds["X_train"].shape[1]
            except:
                n_cols = 1 # Fallback
            
            # If strictly tabular/numpy, get exact size
            if isinstance(ds["X_train"], (pd.DataFrame, np.ndarray)):
                size = ds["X_train"].size
            else:
                size = n_rows * n_cols
                
            if size > MAX_CELLS:
                print(f"  -> Skipping {ds['name']} (Size: {size} > {MAX_CELLS}) to save runtime.")
            else:
                valid_datasets.append(ds)
        
        if not valid_datasets:
            print(f"Error: All datasets in {benchmark} were filtered out by size constraint.")
            continue

        print(f"Selected {len(valid_datasets)} datasets after filtering.")
        
        # 3. Create Tasks
        tasks = [(ds, SELECTED_MODEL, s) for ds in valid_datasets for s in SEEDS]
        
        print(f"Executing GridSearch across {len(tasks)} tasks...")
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(process_benchmark_grid)(*t) for t in tasks
        )

        success_results = [r for r in results if r["status"] == "success"]
        failed_results = [r for r in results if r["status"] == "failed"]
        
        if failed_results:
            print(f"Warning: {len(failed_results)} tasks failed in {benchmark}.")

        df_results = pd.DataFrame(success_results)
        
        if df_results.empty:
            print(f"Error: All tasks failed for {benchmark}. No results to save.")
            continue
            
        # Identify parameter columns (keys that look like dicts, excluding the time_ ones)
        param_cols = [c for c in df_results.columns if c.startswith("{") and not c.startswith("time_")]
        
        # --- Calculate Average Ranks ---
        # For regression, lower score (MAE/RMSE) is better, but scikit-learn uses negative RMSE.
        # Neg RMSE: -5 is "better" than -10. So higher is better (ascending=False).
        # Accuracy: Higher is better (ascending=False).
        # Wait - scikit-learn grid search results are maximized. 
        # But we stored abs(score) for regression in process_benchmark_grid.
        # If regression score is MAE/RMSE (positive), lower is better -> ascending=True
        # If classification score is Acc (positive), higher is better -> ascending=False
        
        is_regression = "regression" in [d["task"] for d in valid_datasets[:1]]
        ascending_rank = True if is_regression else False
        
        # Group by dataset and average over seeds
        df_mean_seeds = df_results.groupby("dataset")[param_cols].mean()
        df_ranks = df_mean_seeds.rank(axis=1, ascending=ascending_rank)
        mean_ranks = df_ranks.mean().sort_values()
        
        best_overall_params = mean_ranks.index[0]
        
        # --- Time Aggregation ---
        total_grid_time_seconds = df_results["time_taken"].sum()
        total_best_param_time_seconds = df_results[f"time_{best_overall_params}"].sum()
        
        print("-" * 50)
        print(f"BENCHMARK COMPLETED: {benchmark} | MODEL: {SELECTED_MODEL}")
        print(f"IDEAL HYPERPARAMETERS: {best_overall_params}")
        print(f"Total GridSearch Time: {total_grid_time_seconds / 60:.2f} minutes")
        print(f"Time if we only ran Best Params: {total_best_param_time_seconds / 60:.2f} minutes")
        print("-" * 50)

        # File saving with timestamp to prevent overwrites
        file_prefix = f"grid_{SELECTED_MODEL}_{benchmark}_{run_timestamp}"
        
        df_results.to_csv(f"results/{file_prefix}_raw.csv", index=False)
        df_ranks.to_csv(f"results/{file_prefix}_ranks.csv")
        
        # Save summary
        with open(f"results/{file_prefix}_ideal_params.txt", "w") as f:
            f.write(f"Benchmark: {benchmark}\n")
            f.write(f"Model: {SELECTED_MODEL}\n")
            f.write(f"Run Timestamp: {run_timestamp}\n")
            f.write(f"Total GridSearch Time: {total_grid_time_seconds / 60:.2f} minutes\n")
            f.write(f"Time if Best Params used: {total_best_param_time_seconds / 60:.2f} minutes\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best Hyperparameters: {best_overall_params}\n\n")
            f.write("All Parameter Average Ranks (Lowest = Best):\n")
            f.write(mean_ranks.to_string())
            
        print(f"Saved {benchmark} cleanly to results directory.\n")

    print("All benchmarks finished successfully.")