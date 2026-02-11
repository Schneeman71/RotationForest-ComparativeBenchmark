# RotationForest-ComparativeBenchmark
This is a repository for the purpose of benchmarking and exploring the implications of the rotation forest ensemble machine learning method. 

We are analyzing the performance of the random forest, extra trees, and rotation forest ensemble methods across 4 benchmarks.
The benchmarks are datasets with predefined train and test sets.

- Openml CC-18 or suite 99 (classification)
- Openml suite 297 (regression)
- Aeon tsc (time series classification)
- Aeon tser (timer series regression)

We are using the aeon rotation forest implementation.

We are using the sklearn extra trees and random forest implementations.

Results are currently conducted across 5 seeds and up to 30 datasets from each benchmark.

Graphs: Graphs are created from the results data. Accuracy is a direct comparison. Due to scale, RMSE is normalized by setting 1 equal to the worst RMSE across all tests on the dataset (regardless of model) and RMSE is appropriately scaled. Thus lower RMSE indicated better preformance. 
