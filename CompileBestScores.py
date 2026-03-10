import os
import glob
import re
import pandas as pd
import functools

def compile_best_scores():
    # Define exact target directory
    results_dir = os.path.join('results', 'Benchtest1')
    output_file = 'consolidated_best_scores.csv'
    
    # Find all csv files in Benchtest1
    search_pattern = os.path.join(results_dir, '*.csv')
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No CSV files found in {results_dir}. Please check the path.")
        return

    model_bench_dict = {}

    for f in files:
        filename = os.path.basename(f)
        
        # Regex to parse 'model-benchmark.csv' (e.g., et-openml-297.csv)
        # Groups: 1=model (everything up to first hyphen), 2=benchmark (everything after)
        match = re.search(r'^([^-]+)-(.*)\.csv$', filename)
            
        if match:
            model = match.group(1).lower()
            benchmark = match.group(2).lower()
        else:
            print(f"Skipping {filename}, doesn't match 'model-benchmark.csv' pattern.")
            continue

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue
        
        # Identify columns that are not scores
        non_score_cols = ['dataset', 'seed', 'time_taken', 'status']
        score_cols = [c for c in df.columns if c not in non_score_cols and not c.startswith('time_')]
        
        # Coerce any string errors to numeric
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
        
        # Average across the 5 seeds for each dataset
        df_grouped = df.groupby('dataset')[score_cols].mean()
        
        # Determine task type to know whether we want the highest or lowest score
        is_minimization = 'tser' in benchmark or '297' in benchmark or 'regression' in benchmark
        task_type = 'regression' if is_minimization else 'classification'
        
        # Select the best hyperparameter per dataset
        if is_minimization:
            best_scores = df_grouped.min(axis=1) # lowest MSE/RMSE
        else:
            best_scores = df_grouped.max(axis=1) # highest Accuracy/F1
            
        # Convert to DataFrame
        res_df = best_scores.reset_index()
        res_df.columns = ['dataset', model]
        res_df['benchmark'] = benchmark
        res_df['task_type'] = task_type
        
        model_bench_dict[(model, benchmark)] = res_df

    if not model_bench_dict:
        print("No valid data processed.")
        return

    # Group dataframes by model
    model_dfs = {}
    for (model, benchmark), df in model_bench_dict.items():
        if model not in model_dfs:
            model_dfs[model] = []
        model_dfs[model].append(df)
        
    final_dfs_to_merge = []
    for model, dfs in model_dfs.items():
        combined_model_df = pd.concat(dfs, ignore_index=True)
        final_dfs_to_merge.append(combined_model_df)
        
    # Merge all models horizontally
    consolidated_df = functools.reduce(
        lambda left, right: pd.merge(left, right, on=['dataset', 'benchmark', 'task_type'], how='outer'), 
        final_dfs_to_merge
    )
    
    # Reorder columns: dataset, benchmark, task_type, model1, model2...
    base_cols = ['dataset', 'benchmark', 'task_type']
    model_cols = [c for c in consolidated_df.columns if c not in base_cols]
    consolidated_df = consolidated_df[base_cols + model_cols]
    
    consolidated_df.to_csv(output_file, index=False)
    print(f"Successfully saved compiled scores to {output_file}!")
    print(consolidated_df.head())

if __name__ == '__main__':
    compile_best_scores()