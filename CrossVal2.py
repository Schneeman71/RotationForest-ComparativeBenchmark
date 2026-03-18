import os
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from PreliminaryTesting import (
    load_openml_suite, load_aeon_suite,
    ensure_numeric_format, make_smart_preprocessor, make_model,
    SEEDS, N_JOBS, MAX_CELLS_PER_DATASET
)

def get_param_grid(model_name):
    """Defines the search space for tuning."""
    if model_name in ["rf", "et"]:
        return {
            "model__max_features": ["sqrt", 0.5, 1.0],
            "model__min_samples_leaf": [1, 5]
        }
    elif model_name == "rotf":
        return {
            "model__n_estimators": [20, 50],
            "model__max_group": [3, 7],
            "model__remove_proportion": [0.25, 0.5, 0.75]
        }
    return {}

def process_benchmark_grid(ds, model_name, seed):
    """Runs GridSearchCV with optimized data handling."""
    try:
        t_start = time.time()
        
        # ensure_numeric_format now uses memory-efficient 'views'
        X_train_orig = ensure_numeric_format(ds["X_train"], ds["data_type"])
        y_train = ds["y_train"]

        # 1. Handle NaNs in Target (Regression)
        if ds["task"] == "regression":
            mask = y_train.notna() if hasattr(y_train, "notna") else ~np.isnan(y_train)
            if not mask.all():
                X_train_orig = X_train_orig[mask]
                y_train = y_train[mask]

        # 2. Label Encoding
        if ds["task"] == "classification":
            y_train = LabelEncoder().fit_transform(y_train)

        # 3. Build Tuned Model Pipeline
        preprocessor = make_smart_preprocessor(X_train_orig, ds["task"])
        base_pipe = make_model(model_name, ds["task"], preprocessor, seed)
        
        # 4. CV Strategy Setup
        n_folds = 3
        if ds["task"] == "classification":
            counts = pd.Series(y_train).value_counts()
            cv_strategy = StratifiedKFold(n_folds, shuffle=True, random_state=seed) if counts.min() >= n_folds else KFold(n_folds, shuffle=True, random_state=seed)
            scoring = "accuracy"
        else:
            cv_strategy = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            scoring = "neg_root_mean_squared_error"

        # 5. GridSearch (n_jobs=1 inside Parallel workers to avoid over-subscription)
        search = GridSearchCV(base_pipe, get_param_grid(model_name), cv=cv_strategy, scoring=scoring, n_jobs=1)
        search.fit(X_train_orig, y_train)

        cv_results = search.cv_results_
        # Added model_name to the record for better CSV tracking
        record = {"dataset": ds["name"], "model": model_name, "seed": seed, "time_taken": round(time.time() - t_start, 4), "status": "success"}
        
        for i, params in enumerate(cv_results["params"]):
            param_str = str(params).replace("model__", "") 
            record[param_str] = abs(cv_results["mean_test_score"][i]) if ds["task"] == "regression" else cv_results["mean_test_score"][i]
            record[f"time_{param_str}"] = round((cv_results["mean_fit_time"][i] + cv_results["mean_score_time"][i]) * n_folds, 4)
            
        return record

    except Exception as e:
        return {"dataset": ds.get("name"), "model": model_name, "seed": seed, "status": "failed", "error": str(e)}

if __name__ == "__main__":
    # Define multiple models here
    SELECTED_MODELS = ["et", "rf"] #rotf 
    BENCHMARKS = ["AEON-TSC", "AEON-TSER"] 
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)

    for benchmark in BENCHMARKS:
        print(f"\n==================================================")
        print(f"Processing Benchmark: {benchmark} (Max Cells: {MAX_CELLS_PER_DATASET:,})")
        print(f"==================================================")
        
        # 1. Load Datasets using updated, metadata-filtered loaders
        if benchmark == "OpenML-CC18":
            datasets = load_openml_suite(99, "classification", "OpenML-CC18")
        elif benchmark == "OpenML-297":
            datasets = load_openml_suite(297, "regression", "OpenML-297")
        elif benchmark == "AEON-TSC":
            datasets = load_aeon_suite("TSC")
        elif benchmark == "AEON-TSER":
            datasets = load_aeon_suite("TSER")
        else:
            datasets = []

        if not datasets: 
            print(f"No datasets loaded for {benchmark}. Skipping...")
            continue
            
        # Iterate through each model for the current benchmark
        for model_name in SELECTED_MODELS:
            print(f"\n---> Evaluating Model: {model_name.upper()} on {benchmark}")
            
            # 2. Parallel Task Execution
            # Build task list strictly for THIS model so dataframe columns align perfectly later
            tasks = [(ds, model_name, s) for ds in datasets for s in SEEDS]
            
            print(f"Executing {len(tasks)} tasks in parallel...")
            results = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(process_benchmark_grid)(*t) for t in tasks)

            # 3. Process & Save Results
            success_results = [r for r in results if r["status"] == "success"]
            if not success_results: 
                print(f"All tasks failed for {model_name} on {benchmark}.")
                continue

            df_results = pd.DataFrame(success_results)
            param_cols = [c for c in df_results.columns if c.startswith("{") and not c.startswith("time_")]
            
            # Summary & Ranking (Calculated per model so grid search spaces don't clash)
            df_mean_seeds = df_results.groupby("dataset")[param_cols].mean()
            is_regression = "TSER" in benchmark or "297" in benchmark
            df_ranks = df_mean_seeds.rank(axis=1, ascending=is_regression)
            mean_ranks = df_ranks.mean().sort_values()
            
            # Save output - file name now includes the specific model being run
            file_prefix = f"grid_{model_name}_{benchmark}_{run_timestamp}"
            df_results.to_csv(f"results/{file_prefix}_raw.csv", index=False)
            print(f"Best Params for {model_name.upper()} on {benchmark}: {mean_ranks.index[0]}")
