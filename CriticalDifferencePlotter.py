import os
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp

# 1. Define paths and groupings
DATA_DIR = os.path.join("results", "Benchtest1")

# The benchmarks and models based on your file naming convention
BENCHMARKS = ["openml-297", "openml-cc18"]
MODELS = ["et", "rf", "rotf"]

def get_best_scores_per_dataset(filepath, is_minimization=False):
    """
    Reads a model's benchmark csv file and extracts the best score.
    If is_minimization=True (Regression/RMSE), it finds the MINIMUM.
    If False (Classification/Accuracy), it finds the MAXIMUM.
    """
    df = pd.read_csv(filepath)
    
    # Identify columns that are NOT scores
    non_score_cols = ['dataset', 'seed', 'time_taken', 'status']
    score_cols = [col for col in df.columns 
                  if col not in non_score_cols and not str(col).startswith('time_')]
    
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
    
    # Average across seeds first
    df_grouped = df.groupby('dataset')[score_cols].mean()
    
    # Adaptive selection: min for error metrics, max for accuracy metrics
    if is_minimization:
        best_scores = df_grouped.min(axis=1)
    else:
        best_scores = df_grouped.max(axis=1)
        
    best_scores.name = 'best_score'
    return best_scores

def generate_cd_plot(benchmark):
    """
    Gathers data and generates CD plot with adaptive ranking logic.
    """
    print(f"Processing Benchmark: {benchmark}...")
    
    # Determine if this benchmark uses error metrics (lower is better)
    # openml-297 is regression (RMSE), openml-cc18 is classification (Accuracy)
    is_minimization = True if "297" in benchmark else False
    
    model_scores = {}
    
    for model in MODELS:
        filename = f"{model}-{benchmark}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"  Warning: File not found -> {filepath}")
            continue
            
        # Pass the minimization flag to the score extractor
        model_scores[model] = get_best_scores_per_dataset(filepath, is_minimization)

    results_df = pd.DataFrame(model_scores).dropna()
    
    if results_df.empty:
        print(f"  Error: No overlapping datasets found for {benchmark}. Skipping.\n")
        return

    # --- ADAPTIVE RANKING LOGIC ---
    # If minimization (RMSE): ascending=True (lowest value gets rank 1)
    # If maximization (Accuracy): ascending=False (highest value gets rank 1)
    rank_ascending = True if is_minimization else False
    ranks_df = results_df.rank(axis=1, ascending=rank_ascending)
    
    avg_ranks = ranks_df.mean()
    print(f"  Metric Type: {'Minimization (Lower is better)' if is_minimization else 'Maximization (Higher is better)'}")
    print(f"  Average Ranks:\n{avg_ranks.to_string()}\n")

    # Generate Plot
    plt.figure(figsize=(8, 3), dpi=150)
    # Add metric type to title for clarity
    metric_label = "RMSE/MSE" if is_minimization else "ACC/F1"
    plt.title(f"CD Plot: {benchmark.upper()} ({metric_label})", pad=20)
    
    # P-values: Nemenyi test
    p_values = sp.posthoc_nemenyi_friedman(results_df)
    p_values.columns = results_df.columns
    p_values.index = results_df.columns
    
    sp.critical_difference_diagram(avg_ranks, p_values)
    
    output_filename = f"CD_plot_{benchmark}.png"
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot as {output_filename}\n")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
    else:
        for benchmark in BENCHMARKS:
            generate_cd_plot(benchmark)
        print("All plots generated successfully!")