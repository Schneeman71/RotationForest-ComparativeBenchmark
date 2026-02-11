import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import numpy as np

# --- Configuration ---
# Path to your result file
input_path = os.path.join('results', 'benchmark_official_v1.csv')
output_csv_name = 'ResultsAveragedAcrossSeeds_Normalized.csv'

# Ensure the file exists
if not os.path.exists(input_path):
    if os.path.exists('benchmark_official_v1.csv'):
        input_path = 'benchmark_official_v1.csv'
    else:
        raise FileNotFoundError(f"Could not find the file at {input_path}")

print(f"Loading data from {input_path}...")
df = pd.read_csv(input_path)

# --- Part 1: Normalization (Scale RMSE to 0-1 per Dataset) ---
print("Normalizing RMSE scores per dataset...")

# --- Corrected Normalization Section ---
print("Normalizing RMSE scores per dataset...")

def normalize_group(group):
    # Determine the metric for this specific group
    metric = group['metric_type'].iloc[0]
    
    if metric == 'RMSE':
        max_val = group['score'].max()
        if max_val != 0:
            group['score'] = group['score'] / max_val
            
    return group

df_normalized = df.groupby(['tablename', 'classification_name'], group_keys=False).apply(normalize_group)

# Apply the normalization function to each dataset separately
df_normalized = df.groupby('tablename', group_keys=False).apply(normalize_group)


# --- Part 2: Aggregation ---
print("Aggregating normalized results across seeds...")

group_cols = [
    'classification_name', 'tablename', 'feature_type', 
    'model_type', 'metric_type', 'dataset_rows', 'dataset_columns'
]

agg_dict = {
    'score': ['mean', 'std'],
    'time_taken': 'mean'
}

# Perform aggregation on the NORMALIZED data
df_averaged = df_normalized.groupby(group_cols).agg(agg_dict).reset_index()

# Flatten columns
df_averaged.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_averaged.columns.values]

# Rename columns
df_averaged.rename(columns={
    'score_mean': 'avg_score',
    'score_std': 'std_score',
    'time_taken_mean': 'avg_time_taken'
}, inplace=True)

# Save the averaged document
df_averaged.to_csv(output_csv_name, index=False)
print(f"Saved normalized & averaged results to: {output_csv_name}")


# --- Part 3: Visualization ---
print("Generating visualizations...")
sns.set(style="whitegrid")

classifications = df_normalized['classification_name'].unique()

for cls_name in classifications:
    print(f"Processing benchmark: {cls_name}")
    
    # Filter data
    cls_data = df_normalized[df_normalized['classification_name'] == cls_name]
    datasets = cls_data['tablename'].unique()
    datasets.sort()
    
    metric_name = cls_data['metric_type'].iloc[0]
    
    # Determine Y-axis label based on normalization
    if metric_name == 'RMSE':
        y_label = 'Normalized RMSE (Relative to Max Error)'
    else:
        y_label = 'Accuracy Score'
    
    # Chunk datasets into groups of 5
    chunk_size = 5
    num_chunks = math.ceil(len(datasets) / chunk_size)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        dataset_chunk = datasets[start_idx:end_idx]
        
        # Filter for the current batch of datasets
        plot_data = cls_data[cls_data['tablename'].isin(dataset_chunk)]
        
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        # Note: 'errorbar' automatically calculates the CI or SD from the raw (normalized) data
        sns.barplot(
            data=plot_data,
            x='tablename',
            y='score',
            hue='model_type',
            errorbar='sd',     
            palette='viridis', 
            capsize=0.1,       
            edgecolor='black', 
            alpha=0.9
        )
        
        plt.title(f'{cls_name} Results - Batch {i+1}/{num_chunks}', fontsize=16, weight='bold')
        plt.ylabel(y_label, fontsize=14)
        plt.xlabel('Dataset', fontsize=14)
        
        # Move legend outside
        plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        
        # Save graph
        safe_cls_name = cls_name.replace(' ', '_')
        filename = f"{safe_cls_name}_Batch_{i+1}_Normalized.png"
        plt.savefig(filename, dpi=300)
        plt.close()

print("All visualizations completed!")