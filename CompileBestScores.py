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
        
        # Calculate BOTH mean and standard deviation across the 5 seeds for each dataset
        df_mean = df.groupby('dataset')[score_cols].mean()
        df_std = df.groupby('dataset')[score_cols].std()
        
        # Determine task type to know whether we want the highest or lowest score
        is_minimization = 'tser' in benchmark or '297' in benchmark or 'regression' in benchmark
        task_type = 'regression' if is_minimization else 'classification'
        
        # Select the best hyperparameter COLUMN per dataset
        if is_minimization:
            best_param_cols = df_mean.idxmin(axis=1) # col name with lowest MSE/RMSE
        else:
            best_param_cols = df_mean.idxmax(axis=1) # col name with highest Accuracy/F1
            
        # Extract the mean and std specifically for that ideal hyperparameter
        best_records = []
        for dataset in df_mean.index:
            best_col = best_param_cols[dataset]
            best_records.append({
                'dataset': dataset,
                f'{model}_mean': df_mean.loc[dataset, best_col],
                f'{model}_std': df_std.loc[dataset, best_col]
            })
            
        # Convert to DataFrame
        res_df = pd.DataFrame(best_records)
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
    
    # Reorder columns: dataset, benchmark, task_type, model1_mean, model1_std, model2_mean...
    base_cols = ['dataset', 'benchmark', 'task_type']
    model_cols = [c for c in consolidated_df.columns if c not in base_cols]
    
    # Sort model columns so that _mean and _std for the same model are next to each other
    model_cols.sort()
    
    consolidated_df = consolidated_df[base_cols + model_cols]
    
    consolidated_df.to_csv(output_file, index=False)
    print(f"Successfully saved compiled scores with standard deviations to {output_file}!")
    print(consolidated_df.head())

if __name__ == '__main__':
    compile_best_scores()