# `run_clustering`

This function performs clustering on a DataFrame containing embedding vectors by orchestrating several components: PCA preprocessing (if enabled), UMAP dimensionality reduction, HDBSCAN clustering (with optional branch detection), and hyperparameter optimization via Optuna. It returns a dictionary with the clustered DataFrame and the final model instances.

## Parameters

- **filtered_df** (`pd.DataFrame`): The input DataFrame containing the embedding vectors. It must include a column defined by `embedding_col_name`.
- **min_clusters** (`int`, default=3): Minimum acceptable number of clusters. Set lower if you expect fewer distinct groups in your data.
- **max_clusters** (`int`, default=26): Maximum acceptable number of clusters. Increase this for highly diverse datasets.
- **trials_per_batch** (`int`, default=10): Number of hyperparameter tuning trials per batch. A higher number reduces the frequency of stopping checks but increases runtime per batch.
- **min_pareto_solutions** (`int`, default=5): Minimum number of Pareto-optimal solutions required to stop the optimization. Increase for a broader search if needed.
- **max_trials** (`int`, default=100): Maximum number of trials to run. This prevents the optimizer from running indefinitely.
- **random_state** (`int`, default=42): Seed for reproducibility across UMAP, PCA, and Optuna.
- **embedding_col_name** (`str`, default="embedding_vector"): Name of the column in `filtered_df` that contains the embedding vectors.
- **min_noise_ratio** (`float`, default=0.03): Lower bound on the allowed noise ratio (points not assigned to a cluster). Use lower values for stricter clustering.
- **max_noise_ratio** (`float`, default=0.35): Upper bound on the allowed noise ratio. Increase if your data is expected to be noisier.
- **optuna_jobs** (`int`, default=-1): Number of parallel jobs to run during optimization. `-1` uses all available processors.

### UMAP Configuration Parameters
- **umap_n_neighbors_min** (`int`, default=2): Lower bound for the number of neighbors in UMAP. Lower values emphasize local structure.
- **umap_n_neighbors_max** (`int`, default=25): Upper bound for neighbors. Higher values capture more global relationships.
- **umap_min_dist_min** (`float`, default=0.0): Lower bound for minimum distance between points in the UMAP space. Smaller values yield tighter clusters.
- **umap_min_dist_max** (`float`, default=0.1): Upper bound for the minimum distance. Increase to force more separation.
- **umap_spread_min** (`float`, default=1.0): Minimum spread of clusters in UMAP; controls how far apart clusters can be.
- **umap_spread_max** (`float`, default=10.0): Maximum spread; higher values produce more dispersed clusters.
- **umap_learning_rate_min** (`float`, default=0.08): Lower bound for UMAP’s learning rate. Lower values slow convergence.
- **umap_learning_rate_max** (`float`, default=1.0): Upper bound for learning rate; higher rates can speed up convergence on larger datasets.
- **umap_min_dims** (`int`, default=2): Minimum number of dimensions to reduce to.
- **umap_max_dims** (`int`, default=20): Maximum number of dimensions. Increase if you need to retain more variance.
- **umap_metric** (`str`, default="cosine"): Distance metric for UMAP; common choices include "cosine" and "euclidean".
- **dims** (`int` or `None`, default=3): Fixed number of UMAP dimensions. If set to `None`, a value is sampled between `umap_min_dims` and `umap_max_dims`.

### HDBSCAN Configuration Parameters
- **hdbscan_min_cluster_size_multiplier_min** (`float`, default=0.005): Lower multiplier for computing the minimum cluster size relative to the total number of data points. Lower values allow smaller clusters.
- **hdbscan_min_cluster_size_multiplier_max** (`float`, default=0.025): Upper multiplier for minimum cluster size.
- **hdbscan_min_samples_min** (`int`, default=2): Lower bound for HDBSCAN’s `min_samples`. Lower values allow more clusters with fewer points.
- **hdbscan_min_samples_max** (`int`, default=50): Upper bound for `min_samples`. Use higher values for requiring denser clusters.
- **hdbscan_epsilon_min** (`float`, default=0.0): Lower bound for the epsilon parameter. Controls cluster sensitivity.
- **hdbscan_epsilon_max** (`float`, default=1.0): Upper bound for epsilon; higher values may merge clusters.
- **hdbscan_metric** (`str`, default="euclidean"): Distance metric for HDBSCAN.
- **hdbscan_cluster_selection_method** (`str`, default="eom"): Cluster selection method (categorical); typically "eom".
- **hdbscan_outlier_threshold** (`int`, default=10): Percentile threshold for determining outliers. Adjust if too many points are classified as noise.

### PCA Configuration
- **target_pca_evr** (`float`, default=0.9): Target explained variance ratio for PCA preprocessing. Higher values preserve more variance but may result in higher-dimensional outputs.

### Branch Detection Configuration
- **hdbscan_branch_detection** (`bool`, default=False): Enable branch detection in HDBSCAN if hierarchical branching is expected.
- **branch_min_cluster_size_multiplier_min** (`float`, default=0.005): Lower multiplier for branch detection’s minimum cluster size.
- **branch_min_cluster_size_multiplier_max** (`float`, default=0.025): Upper multiplier for branch detection.
- **branch_selection_persistence_min** (`float`, default=0.0): Lower bound for branch selection persistence; higher values enforce stronger branch separation.
- **branch_selection_persistence_max** (`float`, default=0.1): Upper bound for branch selection persistence.
- **branch_label_sides_as_branches** (`bool`, default=False): If `True`, treats sides of branches as separate branches.

## Usage Considerations

- **Dataset Size**: Larger datasets may require higher `umap_n_neighbors` and learning rate values. For smaller datasets, lower values are typically more appropriate.
- **Noise Sensitivity**: Adjust `min_noise_ratio` and `max_noise_ratio` based on how much noise you expect in your data.
- **Cluster Granularity**: Set `min_clusters` and `max_clusters` according to the expected number of distinct groups.
- **Hyperparameter Tuning**: The provided ranges allow exploration during optimization; narrow them if you have domain-specific insights.