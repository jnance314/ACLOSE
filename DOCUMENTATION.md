# run_clustering

```python
def run_clustering(
    filtered_df: pd.DataFrame,
    min_clusters=3,
    max_clusters=26,
    trials_per_batch=10,
    min_pareto_solutions=5,
    max_trials=100,
    random_state=42,
    embedding_col_name="embedding_vector",
    min_noise_ratio=0.03,
    max_noise_ratio=0.35,
    optuna_jobs=-1,
    # UMAP configuration parameters:
    umap_n_neighbors_min=2,
    umap_n_neighbors_max=25,
    umap_min_dist_min=0.0,
    umap_min_dist_max=0.1,
    umap_spread_min=1.0,
    umap_spread_max=10.0,
    umap_learning_rate_min=0.08,
    umap_learning_rate_max=1.0,
    umap_min_dims=2,
    umap_max_dims=20,
    umap_metric="cosine",
    dims=3,
    # HDBSCAN configuration parameters:
    hdbscan_min_cluster_size_multiplier_min=0.005,
    hdbscan_min_cluster_size_multiplier_max=0.025,
    hdbscan_min_samples_min=2,
    hdbscan_min_samples_max=50,
    hdbscan_epsilon_min=0.0,
    hdbscan_epsilon_max=1.0,
    hdbscan_metric="euclidean",
    hdbscan_cluster_selection_method="eom",
    hdbscan_outlier_threshold=10,
    # PCA configuration:
    target_pca_evr=0.9,
    # Branch detection configuration:
    hdbscan_branch_detection=False,
    branch_min_cluster_size_multiplier_min=0.005,
    branch_min_cluster_size_multiplier_max=0.025,
    branch_selection_persistence_min=0.0,
    branch_selection_persistence_max=0.1,
    branch_label_sides_as_branches=False,
):
    """
    Perform clustering on a DataFrame containing embedding vectors.

    This function optimizes UMAP and HDBSCAN hyperparameters to find the best clustering solution 
    using a multi-objective approach (silhouette score, noise ratio, and number of clusters).

    Returns a dictionary containing the clustered DataFrame, models used, and metrics.

    Parameters:
        filtered_df (pd.DataFrame): 
            DataFrame containing embedding vectors to cluster. Must have a column named 
            by embedding_col_name (default: "embedding_vector") containing vector data as
            list-like objects (Python lists, numpy arrays, etc.). Each vector should be
            a numerical embedding, typically high-dimensional (e.g., 768 or 1536 dimensions
            from language models like BERT or OpenAI embeddings).

        min_clusters (int, default=3): 
            Minimum acceptable number of clusters. Trials producing fewer clusters will 
            be rejected. Lower values allow more general clustering with fewer topics, 
            while higher values force more granular clustering.

        max_clusters (int, default=26): 
            Maximum acceptable number of clusters. Trials producing more clusters will 
            be rejected. Lower values create broader, more general clusters, while 
            higher values allow for more specific, fine-grained clustering, suitable 
            for larger datasets with diverse content.

        trials_per_batch (int, default=10): 
            Number of hyperparameter optimization trials to run per batch. The optimization
            process runs in batches to allow for early stopping once sufficient Pareto-optimal
            solutions are found. Each trial tests a different combination of hyperparameters
            (UMAP and HDBSCAN settings) and evaluates them using multiple objectives. Higher 
            values increase the chance of finding optimal solutions at the cost of computation 
            time. Consider increasing for complex data or when using powerful hardware.

        min_pareto_solutions (int, default=5): 
            Minimum number of Pareto-optimal solutions to find before stopping optimization.
            A Pareto-optimal solution is one where no objective (silhouette score, negative
            noise ratio, or negative number of clusters) can be improved without sacrificing
            another objective. These represent balanced trade-offs between cluster quality,
            noise level, and number of topics. Higher values ensure a more thorough exploration
            of this trade-off space, but require more computation time. For most applications,
            5-10 is a good balance. The best solution is selected from this Pareto frontier
            using TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

        max_trials (int, default=100): 
            Maximum number of trials to run during optimization. This is a safety limit to 
            prevent infinite loops. Increase for complex datasets or when using many 
            hyperparameter combinations. Stricter filtering parameters (particularly min/max
            clusters and noise ratio constraints) will reject more trials, requiring more
            total trials to find sufficient Pareto-optimal solutions. Similarly, forcing
            very low dimensions (e.g., dims=2) will require more trials to find good
            solutions. If dims=None (allowing dimension optimization), fewer trials are
            typically needed. Decrease this value to limit computation time if needed.

        random_state (int, default=42): 
            Seed for reproducibility across UMAP, PCA, and Optuna. UMAP and other dimensionality
            reduction algorithms have stochastic components that can produce different results
            with each run. Setting a fixed random_state ensures you get consistent results
            when running with identical parameters, which is critical for reproducible research
            and reliable production deployments. Any integer value can be used; changing this
            value will produce different (but still valid) clustering solutions.

        embedding_col_name (str, default="embedding_vector"): 
            Name of the column in filtered_df that contains the embedding vectors. These should
            be list-like objects containing numerical values (e.g., [0.123, -0.456, ...]) from
            a language model or other embedding technique. The vectors must all have the same
            dimensionality. Change this parameter if your DataFrame uses a different column
            name for the embeddings.

        min_noise_ratio (float, default=0.03): 
            Minimum acceptable noise ratio (proportion of points classified as noise). 
            Lower values force more points into clusters, which can lead to less coherent 
            clusters. For clean, well-separated data, values as low as 0.01 may work well.

        max_noise_ratio (float, default=0.35): 
            Maximum acceptable noise ratio. Higher values allow more points to be classified 
            as noise, potentially leading to more coherent clusters but less coverage. For 
            noisy data with outliers, values up to 0.5 might be appropriate.

        optuna_jobs (int, default=-1): 
            Number of parallel jobs for Optuna optimization. -1 uses all available cores. 
            Higher values can significantly speed up optimization on multi-core systems but 
            increases memory usage. You might want to use a specific number (e.g., 4 or 8)
            rather than -1 when running in resource-constrained environments (like cloud 
            functions, Docker containers, or shared servers), when you want to leave some 
            CPU cores available for other processes, or when memory is limited (as each 
            worker requires its own memory allocation).

        # UMAP Parameters (all are search bounds for hyperparameter optimization)

        umap_n_neighbors_min (int, default=2), umap_n_neighbors_max (int, default=25): 
            Bounds for UMAP's n_neighbors parameter, which controls the balance between local 
            and global structure. Lower values (2-5) preserve local structure but may fragment 
            clusters, suitable for finding fine-grained patterns. Higher values (15-50) preserve 
            global structure, better for general topic separation. For most text clustering, 
            10-20 works well.

        umap_min_dist_min (float, default=0.0), umap_min_dist_max (float, default=0.1): 
            Bounds for UMAP's min_dist parameter, which controls how tightly points cluster. 
            Lower values (0.0-0.1) create tighter, more compact clusters, good for finding 
            distinct groups. Higher values (0.5-1.0) create more evenly dispersed embeddings, 
            useful when clusters might overlap. Most text clustering works well with values 
            under 0.2.

        umap_spread_min (float, default=1.0), umap_spread_max (float, default=10.0): 
            Bounds for UMAP's spread parameter, which affects the scale of the embedding. 
            Lower values create a more compressed visualization, while higher values spread 
            points out more. This primarily affects visualization rather than clustering quality.

        umap_learning_rate_min (float, default=0.08), umap_learning_rate_max (float, default=1.0): 
            Bounds for UMAP's learning_rate parameter, which controls the embedding optimization. 
            Lower values produce more stable results but may converge to suboptimal solutions. 
            Higher values explore more of the space but might be less stable.

        umap_min_dims (int, default=2), umap_max_dims (int, default=20): 
            Bounds for UMAP's output dimensionality when dims is None. Lower dimensions (2-5) 
            are easier to visualize but may lose information. Higher dimensions (10-50) preserve 
            more structure but increase computational cost. For clustering without visualization, 
            3-15 dimensions often work well.

        umap_metric (str, default="cosine"): 
            Distance metric for UMAP. Valid options include: "cosine", "euclidean", "manhattan",
            "chebyshev", "minkowski", "canberra", "braycurtis", "mahalanobis", "wminkowski",
            "seuclidean", "correlation", and "haversine". "cosine" is generally best for text
            embeddings as it focuses on direction rather than magnitude. "euclidean" is better
            for normalized embeddings and becomes mathematically equivalent to cosine distance
            when vectors are normalized. "manhattan" can be more robust to outliers.

        dims (int, default=3): 
            Fixed dimensionality for UMAP reduction. If provided, this overrides the min/max dims 
            search. Set to None to allow optimization to search for the best dimensionality. 
            3 is good for visualization, while higher values (5-15) might create better clusters 
            for complex datasets.

        # HDBSCAN Parameters

        hdbscan_min_cluster_size_multiplier_min (float, default=0.005), 
        hdbscan_min_cluster_size_multiplier_max (float, default=0.025): 
            Bounds for calculating HDBSCAN's min_cluster_size as a fraction of the dataset size. 
            Lower values (0.001-0.01) allow smaller clusters, good for finding niche topics in 
            diverse data. Higher values (0.02-0.1) require larger, more significant clusters, 
            better for identifying major themes. For a dataset of 1000 points, these defaults 
            would search min_cluster_size between 5 and 25.

        hdbscan_min_samples_min (int, default=2), hdbscan_min_samples_max (int, default=50): 
            Bounds for HDBSCAN's min_samples parameter, which determines how conservative the 
            clustering is. Lower values are more aggressive in forming clusters, while higher 
            values require more evidence for cluster membership, producing more robust clusters 
            but potentially more noise.

        hdbscan_epsilon_min (float, default=0.0), hdbscan_epsilon_max (float, default=1.0): 
            Bounds for HDBSCAN's epsilon parameter, which allows relaxed cluster membership. 
            Higher values expand clusters to include more borderline points, reducing noise. 
            Lower values maintain stricter cluster boundaries. 0.0 disables this relaxation.

        hdbscan_metric (str, default="euclidean"): 
            Distance metric for HDBSCAN. Valid options include: "euclidean", "manhattan", 
            "chebyshev", "minkowski", "canberra", "braycurtis", "mahalanobis", "wminkowski",
            "seuclidean", "correlation", and "haversine". "euclidean" works well on UMAP-reduced
            data. Note that for low-dimensional space (after UMAP reduction), euclidean distance
            is typically more appropriate, even if you used cosine distance in UMAP.

        hdbscan_cluster_selection_method (str, default="eom"): 
            Method for cluster extraction in HDBSCAN. Valid options are "eom" or "leaf".
            "eom" (Excess of Mass) tends to produce more clusters of varying sizes, while 
            "leaf" produces more homogeneously sized clusters. For text clustering, "eom" 
            usually provides more meaningful distinctions.

        hdbscan_outlier_threshold (int, default=10): 
            Percentile threshold for determining core points within clusters. Lower values (5-10) 
            are more selective, designating fewer points as core, while higher values (20-30) 
            include more points as core. This affects labeling quality: too low can cause clusters 
            without core points, while too high may include less representative points.

        # PCA Configuration

        target_pca_evr (float, default=0.9): 
            Target explained variance ratio for optional PCA preprocessing. PCA (Principal Component
            Analysis) is applied before UMAP to reduce high-dimensional embeddings by identifying
            the components that contribute the most information. This preprocessing step helps
            UMAP find better manifold embeddings by removing noise and focusing on significant
            patterns in the data. Higher values (0.95-0.99) preserve more information but reduce
            dimensionality less. Lower values (0.7-0.85) aggressively reduce dimensions, potentially
            improving clustering speed and noise reduction at the cost of information loss.
            0.9 is a good balance for most applications. Must be between 0.0 (exclusive) and 1.0 (inclusive).

        # Branch Detection Configuration

        hdbscan_branch_detection (bool, default=False): 
            Whether to enable branch detection in HDBSCAN. When True, identifies branches in the 
            cluster hierarchy, which can reveal sub-topics or hierarchical structure. This can 
            create more nuanced clustering but increases complexity and computation time.

        branch_min_cluster_size_multiplier_min (float, default=0.005), 
        branch_min_cluster_size_multiplier_max (float, default=0.025): 
            Bounds for the branch min_cluster_size multiplier, similar to the HDBSCAN parameter. 
            Only relevant when branch detection is enabled.

        branch_selection_persistence_min (float, default=0.0), 
        branch_selection_persistence_max (float, default=0.1): 
            Bounds for branch selection persistence, which controls how aggressively branches 
            are selected. Higher values require more significant branches.

        branch_label_sides_as_branches (bool, default=False): 
            Whether to label sides as branches in the branch detection process. When True, 
            this can reveal more subtle branching structures.

    Returns:
        dict: A dictionary containing:
            - 'clustered_df' (pd.DataFrame): The input DataFrame augmented with clustering results.
              Added columns include:
                * 'membership_strength' (float): Indicates how strongly each point belongs to 
                  its assigned cluster (higher values = stronger membership). Useful for 
                  filtering points by confidence or identifying borderline cases.
                * 'core_point' (bool): Flag indicating whether the point is a core point of 
                  the cluster (True) or a peripheral point (False). Core points are essential 
                  for later labeling with add_labels().
                * 'outlier_score' (float): For non-branch detection, indicates how outlier-like 
                  a point is (higher = more outlier-like). NaN for branch detection mode.
                * 'reduced_vector' (list): The reduced-dimensional coordinates for each point, 
                  useful for visualization and further analysis.
                * 'cluster_id' (int): The assigned cluster ID, with -1 indicating noise points.
                  This is your primary column for grouping and analyzing clusters.
                  
            - 'umap_model' (umap.UMAP): The fitted UMAP model instance used for dimensionality 
              reduction. Can be used for projecting new data points onto the same embedding space
              with umap_model.transform(new_data).
              
            - 'hdbscan_model' (hdbscan.HDBSCAN): The fitted HDBSCAN model instance used for 
              clustering. Can be used to predict cluster membership for new data points with
              hdbscan_model.approximate_predict(new_data).
              
            - 'pca_model' (sklearn.decomposition.PCA or None): The PCA model instance used for
              preprocessing, or None if PCA was not applied. If present, new data should be
              preprocessed with this model before UMAP projection.
              
            - 'metrics_dict' (dict): Dictionary containing key clustering metrics:
                * 'reduced_dimensions' (int): Final dimensionality used for clustering
                * 'n_clusters' (int): Number of clusters found (excluding noise)
                * 'noise_ratio' (float): Proportion of points classified as noise
                * 'silhouette_score' (float, optional): Average silhouette score of the clustering
              These metrics help evaluate the quality of the clustering and can guide parameter
              adjustments for future runs.
              
            - 'branch_detector' (hdbscan.BranchDetector or None): The fitted BranchDetector if
              branch detection was enabled, else None. Used internally but also available if
              you need to apply branch detection logic to new data.
    """
```

# add_labels

```python
def add_labels(
    cluster_df: pd.DataFrame,
    llm_model: str = "o1",
    language: str = "english",
    temperature: float = 1.0,
    data_description: str = "No data description provided. Just do your best to infer/assume the context of the data while performing your tasks.",
    ascending: bool = False,
    core_top_n: int = 10,
    peripheral_n: int = 12,
    num_strata: int = 3,
    content_col_name: str = "content",
) -> Dict[str, object]:
    """
    Generate semantic topic labels for clusters using a language model.
    
    This function uses a two-pass approach: first generating initial topic labels from core points,
    then refining them by considering peripheral points. This produces more representative and
    generalizable labels for each cluster.
    
    Parameters:
        cluster_df (pd.DataFrame): 
            DataFrame containing clustering results from run_clustering(). Must include columns:
            'cluster_id' (int): cluster identifiers with -1 for noise
            'core_point' (bool): flags indicating core points within clusters
            'membership_strength' (float): strength of point's association with its cluster
            Also requires a text content column (specified by content_col_name) containing
            the textual data (str) to be used for generating labels.
        
        llm_model (str, default="o1"): 
            Identifier for the language model to use. Valid options are limited to:
            - "o1"
            - "o1-preview"
            - "o1-mini"
            - "o3-mini"
            - "o3-mini-high"
            Choose more capable models for complex data or when high-quality labels are critical.
            
        language (str, default="english"): 
            Language for output labels. Set to the appropriate language if your data is not in
            English (e.g., "spanish", "french", "german", "chinese", etc.). The LLM will generate
            labels in this language.
            
        temperature (float, default=1.0): 
            Controls randomness in the language model. Lower values (0.0-0.5) produce more 
            consistent, deterministic labels. Higher values (0.7-1.0) produce more diverse and
            creative labels. For scientific or technical data, consider lower values; for
            creative content, higher values may be appropriate.
            
        data_description (str, default="..."): 
            Description of the data to provide context to the language model. This text is
            directly included in the prompt sent to the LLM, so you can be as verbose as needed.
            You can use f-strings to dynamically generate context based on your dataset. A good
            description significantly improves label quality by helping the model understand
            domain-specific terminology and concepts. Include information about the data source,
            domain, typical content patterns, and any specific labeling preferences.
            
        ascending (bool, default=False): 
            Order for processing clusters. When False (default), larger clusters are processed
            first. Larger clusters often contain more semantic variance (points more spread out
            in the embedding space), so processing them first helps establish more generalized
            topic labels that are clearly differentiated. When True, smaller clusters are
            processed first, which can be beneficial when smaller clusters represent specific
            niche topics that need precise differentiation from the broader topics in larger
            clusters. This parameter affects the quality of the generated labels, not just
            processing efficiency.
            
        core_top_n (int, default=10): 
            Number of top core points to consider for initial labeling. Higher values (15-20)
            provide more context but increase API costs. Lower values (5-8) are more economical
            but may produce less representative labels. 10 is a good balance for most applications.
            For very diverse clusters, consider increasing.
            
        peripheral_n (int, default=12): 
            Number of peripheral points to sample for label refinement. Higher values give a
            more complete picture of the cluster's diversity but increase API costs. Lower
            values are more economical but may miss important variations. For heterogeneous
            clusters, consider increasing.
            
        num_strata (int, default=3): 
            Number of strata for sampling peripheral points. This divides the non-core points
            into quantiles based on membership strength, ensuring samples represent the full
            spectrum of the cluster's periphery. Each stratum contains points at different
            "distances" from the cluster core in semantic space. Higher values enable more
            nuanced sampling across this distribution, capturing points that are only weakly
            connected to the cluster as well as those just short of being core points.
            For large clusters with varying membership strengths, 3-5 strata work well.
            For smaller clusters, 2-3 is usually sufficient.
            
        content_col_name (str, default="content"): 
            Name of the column in cluster_df containing the text content to be labeled.
            Change this if your DataFrame uses a different column name for the text data.
    
    Returns:
        Dict[str, object]: A dictionary containing:
            - 'dataframe' (pd.DataFrame): The input DataFrame with an added 'topic' column (str)
              containing semantic labels for each data point. This column preserves the original
              structure of your data while adding the generated topic labels. Points from the
              same cluster share the same label. Noise points (cluster_id = -1) are consistently
              labeled as "Noise". If labeling fails for a cluster, it will receive a fallback
              label in the format "Unlabeled_{cluster_id}".
              
            - 'labels_dict' (dict): A dictionary mapping cluster IDs (int) to their topic labels (str).
              This mapping excludes noise points and provides a convenient way to:
                * Get a quick overview of all topics without examining the full DataFrame
                * Map topics to clusters in visualizations or reports
                * Apply the same labels to new data points after clustering
                * Compare topic distributions across different clustering runs
              
              Format: {cluster_id: "Topic Label", ...}
              Example: {0: "Financial News", 1: "Technology Reviews", 2: "Sports Coverage"}
              
    Usage examples:
        # Get the labeled DataFrame for further analysis
        labeled_df = result["dataframe"]
        
        # Count documents per topic
        topic_counts = labeled_df["topic"].value_counts()
        
        # Access just the mapping between cluster IDs and topics
        topic_mapping = result["labels_dict"]
        
        # Use the mapping to create a readable report of cluster sizes
        for cluster_id, topic in topic_mapping.items():
            size = (labeled_df["cluster_id"] == cluster_id).sum()
            print(f"Cluster {cluster_id} ({topic}): {size} documents")
    
    Notes:
        - Requires environment variables OPENAI_API_KEY and HELICONE_API_KEY to be set.
        - Noise points (cluster_id = -1) are automatically labeled as "Noise".
        - The function processes clusters asynchronously for efficiency.
        - The two-pass approach ensures labels are both representative of core cluster concepts
          and generalizable to the entire cluster.
    """
```

# silhouette_fig

```python
def silhouette_fig(clustered_df: pd.DataFrame) -> go.Figure:
    """
    Generate an enhanced silhouette plot for evaluating clustering quality.
    
    The silhouette plot is a powerful visualization that shows how well each point lies within
    its assigned cluster. It helps identify:
    - Well-formed clusters (high silhouette values)
    - Potential misclassifications (negative or low silhouette values)
    - The overall quality of the clustering (average silhouette score)
    
    This function creates a visually enhanced silhouette plot with clusters ordered by size,
    color-coded for easy identification, and includes the average silhouette score as a
    reference line.
    
    Parameters:
        clustered_df (pd.DataFrame): 
            DataFrame containing clustering results from run_clustering(). Must include columns:
            'cluster_id' (int): cluster identifiers with -1 for noise
            'reduced_vector' (list-like): reduced dimensional representations of each point,
                                          typically 2D or 3D coordinates from UMAP reduction
            Silhouette scores are computed using Euclidean distance on these reduced vectors.
    
    Returns:
        go.Figure: 
            A Plotly Figure object containing the silhouette plot visualization. This
            interactive figure includes:
            
            - Individual silhouette profiles for each cluster, color-coded to match
              other visualizations from this package
            - Clusters ordered by size (largest at top) for easier interpretation
            - Text labels identifying each cluster
            - A vertical dashed red line showing the average silhouette score
            - Dark theme styling for better visibility of silhouette patterns
            
            The figure can be:
            - Displayed directly with fig.show()
            - Saved as an HTML file with fig.write_html("silhouette.html")
            - Converted to static images with fig.write_image("silhouette.png")
            - Customized further using Plotly's update_layout() and other methods
            - Embedded in Jupyter notebooks, dashboards, or web applications
            
            The silhouette values range from -1 to 1:
            - Values near 1 indicate points well-matched to their clusters
            - Values near 0 indicate points on cluster boundaries
            - Negative values suggest potential misclassifications
            
            When analyzing this plot, pay attention to:
            - The width of each cluster's profile (indicates cluster size)
            - The shape of each profile (consistent high values indicate coherent clusters)
            - Clusters with many negative values (may indicate fragmented clusters)
            - The average silhouette score (higher is better, >0.5 is generally good)
    
    Notes:
        - Noise points (cluster_id = -1) are excluded from the silhouette calculation.
        - The function automatically assigns colors to clusters for visual distinction.
        - Higher silhouette values (closer to 1.0) indicate better clustering.
        - Average silhouette scores above 0.5 generally indicate reasonable clustering;
          scores above 0.7 indicate strong clustering structure.
        - The plot arranges clusters by size (largest first) to make the visualization
          more interpretable.
        - Clusters with many points near or below 0 may benefit from re-clustering or
          parameter adjustments.
    
    Example interpretation:
        - If most clusters show high silhouette values (>0.5), the clustering is robust.
        - If specific clusters show poor silhouette values, consider adjusting parameters
          or removing those clusters.
        - If the average silhouette is low (<0.3), consider different clustering parameters
          or preprocessing steps.
    """
```

# scatter_fig

```python
def scatter_fig(
    clusters_df: pd.DataFrame,
    content_col_name: str = "content",
    wrap_width: int = 100,
    id_col_name=None,
) -> go.Figure:
    """
    Generate an interactive scatter plot visualization of clustering results.
    
    This function creates a visually rich 2D or 3D plot (depending on the dimensionality of
    the reduced vectors) that shows the spatial distribution of clusters. It's a powerful
    tool for exploring clustering results, inspecting individual data points, and understanding
    the relationships between clusters.
    
    The plot includes:
    - Color-coded points for each cluster
    - Cluster labels positioned at cluster centroids
    - Interactive hover information showing content and metadata for each point
    - Noise points (if any) shown in a distinct color with reduced opacity
    
    Parameters:
        clusters_df (pd.DataFrame): 
            DataFrame containing clustering results from run_clustering(). Must include columns:
            'cluster_id' (int): cluster identifiers with -1 for noise
            'reduced_vector' (list-like): reduced dimensional coordinates, must be 2D or 3D
                                          (e.g., [x, y] or [x, y, z])
            Also requires a content column (specified by content_col_name) containing text (str).
            If 'topic' column (str) exists, it will be used for cluster labels.
        
        content_col_name (str, default="content"): 
            Name of the column containing text content that will be displayed when hovering
            over points in the scatter plot.
        
        wrap_width (int, default=100): 
            Maximum width (in characters) for wrapping text in hover labels. 
            Values below 20 will be automatically increased to 20.
        
        id_col_name (str, default=None): 
            Optional column name for record IDs to include in hover information. For example,
            if your dataset contains movie descriptions, you might use the movie title column
            (e.g., id_col_name="title") to help identify each point in the visualization.
            This can be useful for tracking specific points of interest or connecting
            visualization data back to your original dataset.
    
    Returns:
        go.Figure: 
            A Plotly Figure object containing an interactive scatter plot visualization.
            The function automatically creates either a 2D or 3D plot based on the
            dimensionality of the reduced vectors (2D for 2-dimensional vectors, 3D for
            3-dimensional vectors). For vectors with more than 3 dimensions, a warning
            message is displayed instead.
            
            The figure includes:
            - Color-coded points for each cluster, matching colors used in other visualizations
              from this package
            - Large text labels at cluster centroids showing cluster topics
            - Noise points (if any) displayed with reduced opacity
            - Interactive hover information displaying:
                * Cluster ID and topic (if available)
                * ID value (if id_col_name was provided)
                * Text content from the specified content column
            - A legend for identifying clusters
            - Dark theme styling for better visibility of cluster patterns
            
            The figure can be:
            - Displayed directly with fig.show()
            - Saved as an HTML file with fig.write_html("clusters.html")
            - Converted to static images with fig.write_image("clusters.png")
            - Customized further using Plotly's update_layout() and other methods
            - Embedded in Jupyter notebooks, dashboards, or web applications
            
            For 3D plots, additional interaction is available:
            - Rotation (click and drag)
            - Zoom (scroll)
            - Pan (right-click and drag)
            
            Note that for very large datasets (10,000+ points), the interactive performance 
            may decrease. Consider sampling your data or using a static image export if needed.
    
    Notes:
        - The function automatically determines whether to create a 2D or 3D visualization
          based on the dimensionality of the 'reduced_vector' column.
        - For vectors with more than 3 dimensions, a warning is displayed (visualization
          is limited to 3D).
        - Noise points (cluster_id = -1) are shown with reduced opacity.
        - Hover information includes cluster ID, topic (if available), content, and optional ID.
        - The plot uses a dark theme optimized for visualizing clusters.
        - For large datasets, rendering may be slow. Consider sampling your data first
          if performance is an issue.
    
    Visualization tips:
        - Look for well-separated clusters, which indicate distinct topics
        - Clusters that overlap might benefit from parameter tuning
        - Examine outliers and noise points to identify potential improvements
        - Use the interactive features (zoom, rotation in 3D) to explore the structure
        - Hover over points to inspect individual content and verify cluster coherence
    """
```

# bars_fig

```python
def bars_fig(clusters_df: pd.DataFrame) -> go.Figure:
    """
    Generate a horizontal bar chart visualizing cluster sizes with topic labels.
    
    This function creates an informative bar chart that shows the relative sizes of each cluster,
    identified by both cluster ID and topic label (if available). It provides a quick overview
    of your clustering results, highlighting dominant topics and the distribution of data points
    across clusters.
    
    The visualization:
    - Displays clusters in order of size (smallest to largest by default)
    - Color-codes bars to match other visualizations (silhouette_fig and scatter_fig)
    - Shows exact counts as text annotations
    - Includes both cluster IDs and topic labels for clear identification
    
    Parameters:
        clusters_df (pd.DataFrame): 
            DataFrame containing clustering results from run_clustering(). Must include a
            'cluster_id' column (int) with cluster identifiers (-1 for noise). If a 'topic' 
            column (str) exists (from add_labels), these topics will be displayed alongside 
            cluster IDs in the visualization.
    
    Returns:
        go.Figure: 
            A Plotly Figure object containing a horizontal bar chart visualization.
            The chart provides a clear overview of cluster sizes and topics, with:
            
            - Horizontal bars representing each cluster, sorted by size (smallest to largest)
            - Color-coded bars matching colors used in other visualizations from this package
            - Text labels showing both cluster IDs and topic names
            - Count values displayed at the end of each bar
            - Dark theme styling for better visibility
            - Automatic height scaling based on the number of clusters
            
            The figure can be:
            - Displayed directly with fig.show()
            - Saved as an HTML file with fig.write_html("cluster_sizes.html")
            - Converted to static images with fig.write_image("cluster_sizes.png")
            - Customized further using Plotly's update_layout() and other methods
            - Embedded in Jupyter notebooks, dashboards, or web applications
            
            This visualization complements the scatter_fig() and silhouette_fig() by providing:
            - A quick summary of the cluster distribution in your dataset
            - A clear view of the relative sizes of each topic
            - An easy way to identify dominant vs. niche topics
            - A legend that pairs cluster IDs with their semantic labels
            
            For presentations and reports, consider placing this chart alongside the other
            visualizations to give viewers a complete picture of your clustering results.
    
    Notes:
        - Noise points (cluster_id = -1) are excluded from the visualization to focus on
          meaningful clusters.
        - The chart height automatically scales based on the number of clusters for optimal
          readability.
        - The consistent color coding across visualizations makes it easier to correlate
          information between different views of your data.
        - If the 'topic' column is missing, the function will create default labels in the
          format "Cluster X".
    
    Applications:
        - Quickly assess the distribution of data across topics
        - Identify dominant vs. niche topics
        - Communicate clustering results to stakeholders
        - Track changes in topic distribution when comparing different clustering runs
        - Provide context for more detailed visualizations
    """
```