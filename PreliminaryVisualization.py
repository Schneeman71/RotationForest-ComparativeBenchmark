import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COMPILED_FILE = "consolidated_best_scores.csv"

def generate_time_ranking_plots():
    if not os.path.exists(COMPILED_FILE):
        print(f"Error: Compiled file '{COMPILED_FILE}' not found.")
        return

    df = pd.read_csv(COMPILED_FILE)
    benchmarks = df['benchmark'].unique()
    
    # Set a nice style using seaborn defaults
    sns.set_theme(style="whitegrid")

    for bench in benchmarks:
        print(f"Processing Time Ranking Plot for: {bench}...")
        
        bench_df = df[df['benchmark'] == bench].copy()
        if bench_df.empty:
            continue
            
        # 1. Extract and Calculate Average Time
        time_cols = [c for c in bench_df.columns if c.endswith('_time')]
        
        if not time_cols:
            print(f"  No time columns found for {bench}. Skipping.")
            continue
            
        time_df = bench_df.set_index('dataset')[time_cols]
        # Clean up column names to just have the model name
        time_df.columns = [c.replace('_time', '') for c in time_df.columns]
        
        # Drop rows with NaNs so we rank time fairly across overlapping datasets
        time_df = time_df.dropna(axis=0, how='any')
        
        if time_df.empty:
            print(f"  Not enough overlapping time data for {bench}. Skipping.")
            continue
            
        avg_times = time_df.mean()
        
        # 2. Sort the models by time (fastest to slowest)
        avg_times = avg_times.sort_values(ascending=True)
        
        # 3. Generate the Bar Chart
        plt.figure(figsize=(10, 6), dpi=150)
        
        colors = sns.color_palette("husl", len(avg_times))
        
        # Plot bars
        bars = plt.bar(
            range(len(avg_times)), 
            avg_times.values, 
            color=colors,
            edgecolor='black',
            zorder=3     # Keep bars above the gridlines
        )
        
        # 4. Format X-axis to show Model Name + Time underneath the bar
        tick_labels = [f"{model.upper()}\n({time:.2f}s)" for model, time in avg_times.items()]
        plt.xticks(range(len(avg_times)), tick_labels, fontsize=10, fontweight='bold')
        
        # Optional: Add the exact time floating just above the bar for extra clarity
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                yval, 
                f'{yval:.1f}s', 
                ha='center', 
                va='bottom', 
                fontsize=9,
                color='black'
            )
        
        # Note: Removed log scale from previous version as linear scale 
        # is generally better for standard bar charts unless the variance is astronomical.
        # If one model takes 10 seconds and another takes 3 hours, uncomment the line below:
        # plt.yscale('log')
        
        plt.title(f"Average Execution Time per Dataset: {bench.upper()}", fontsize=14, pad=15)
        plt.xlabel("Models Ranked by Speed (Fastest to Slowest)", fontsize=12)
        plt.ylabel("Average Time (Seconds)", fontsize=12)
        
        # Customize Grid (only horizontal lines needed for bar charts)
        plt.grid(True, axis="y", ls="--", alpha=0.5, zorder=0)
        
        # Save Plot
        output_filename = f"Time_Ranking_Plot_{bench}.png"
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"  Saved plot to {output_filename}")
        plt.close()

if __name__ == "__main__":
    generate_time_ranking_plots()