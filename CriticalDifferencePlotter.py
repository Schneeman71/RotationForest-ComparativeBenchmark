import os
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp

COMPILED_FILE = "consolidated_best_scores.csv"

# ONLY process complete collections (Fixed: Hyphens changed to underscores)
TARGET_BENCHMARKS = [
    "openml_297", 
    "openml_334", 
    "openml_335", 
    "openml_336", 
    "openml_cc18", 
    "aeon_tsc", 
    "aeon_tser"
]

def generate_cd_plot(benchmark, df_benchmark, task_type):
    """
    Generates CD plot using the compiled dataset for a specific benchmark.
    """
    print(f"\nProcessing Benchmark: {benchmark} ({task_type})...")
    
    is_minimization = (task_type == 'regression')
    
    # --- FIX: Only extract the '_mean' columns for the statistical ranking ---
    # We ignore the '_std' columns here because CD plots only rank the primary metric
    mean_cols = [c for c in df_benchmark.columns if c.endswith('_mean')]
    
    # Set index and filter to only our mean score columns
    models_df = df_benchmark.set_index('dataset')[mean_cols]
    
    # Rename columns to remove '_mean' for cleaner plot labels (e.g., 'rf_mean' -> 'RF')
    models_df.columns = [c.replace('_mean', '').upper() for c in models_df.columns]
    
    # Drop models that have completely empty columns for this benchmark
    models_df = models_df.dropna(axis=1, how='all')
    
    # Drop datasets (rows) that are missing values for ANY of the active models
    models_df = models_df.dropna(axis=0, how='any')

    if models_df.empty or models_df.shape[1] < 2:
        print(f"  Error: Not enough overlapping model data for {benchmark}. Skipping.")
        return

    # --- ADAPTIVE RANKING LOGIC ---
    # Minimization (RMSE): ascending=True (lowest error gets rank 1)
    # Maximization (Accuracy): ascending=False (highest accuracy gets rank 1)
    rank_ascending = True if is_minimization else False
    ranks_df = models_df.rank(axis=1, ascending=rank_ascending)
    
    avg_ranks = ranks_df.mean()
    print(f"  Metric Type: {'Minimization (Lower is better)' if is_minimization else 'Maximization (Higher is better)'}")
    print(f"  Average Ranks:\n{avg_ranks.to_string()}\n")

    # Generate Plot
    plt.figure(figsize=(8, 3), dpi=150)
    metric_label = "RMSE/MSE" if is_minimization else "ACC/F1"
    plt.title(f"CD Plot: {benchmark.upper()} ({metric_label})", pad=20)
    
    try:
        # P-values: Nemenyi test
        p_values = sp.posthoc_nemenyi_friedman(models_df)
        
        # Create the diagram
        sp.critical_difference_diagram(avg_ranks, p_values)
        
        output_filename = f"CD_plot_{benchmark}.png"
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"  Saved plot as {output_filename}")
    except Exception as e:
        print(f"  Error generating plot for {benchmark}: {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    if not os.path.exists(COMPILED_FILE):
        print(f"Error: Compiled file '{COMPILED_FILE}' not found.")
        print("Please run 'CompileBestScores.py' first.")
    else:
        df = pd.read_csv(COMPILED_FILE)
        
        # Iterate only over our targeted complete collections
        for benchmark in TARGET_BENCHMARKS:
            if benchmark not in df['benchmark'].values:
                print(f"Warning: Data for '{benchmark}' not found in the consolidated CSV.")
                continue
                
            # Filter for this benchmark
            df_bench = df[df['benchmark'] == benchmark].copy()
            
            # Extract task type (should be identical for all rows of a single benchmark)
            task_type = df_bench['task_type'].iloc[0]
            
            generate_cd_plot(benchmark, df_bench, task_type)
            
        print("\nTargeted plots generated successfully!")