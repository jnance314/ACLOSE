import pandas as pd
import numpy as np
import umap
import hdbscan
from hdbscan import BranchDetector
import math
import optuna
import logging
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from dataclasses import dataclass

# ----------------------
# Configuration Dataclasses
# ----------------------


@dataclass
class UMAPConfig:
    """
    Configuration parameters for UMAP dimensionality reduction.
    """

    n_neighbors_min: int = 2
    n_neighbors_max: int = 25
    min_dist_min: float = 0.0
    min_dist_max: float = 0.1
    spread_min: float = 1.0
    spread_max: float = 10.0
    learning_rate_min: float = 0.08
    learning_rate_max: float = 1.0
    min_dims: int = 2
    max_dims: int = 20
    metric: str = "cosine"
    dims: int | None = 3  # If None, sample from [min_dims, max_dims].


@dataclass
class HDBSCANConfig:
    """
    Configuration parameters for HDBSCAN clustering.
    """

    min_cluster_size_multiplier_min: float = 0.005
    min_cluster_size_multiplier_max: float = 0.025
    min_samples_min: int = 2
    min_samples_max: int = 50
    epsilon_min: float = 0.0
    epsilon_max: float = 1.0
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    outlier_threshold: int = 10  # Percentile threshold for outlier detection.


@dataclass
class BranchDetectionConfig:
    """
    Configuration parameters for branch detection in HDBSCAN.
    The branch min_cluster_size is now defined as a multiplier relative to the number of data points.
    """

    enabled: bool = False
    min_cluster_size_multiplier_min: float = 0.005
    min_cluster_size_multiplier_max: float = 0.025
    selection_persistence_min: float = 0.0
    selection_persistence_max: float = 0.1
    label_sides_as_branches: bool = False


@dataclass
class PCAConfig:
    """
    Configuration parameters for PCA preprocessing.
    """

    target_evr: float | None = 0.90  # Target explained variance ratio.


# ----------------------
# ClusteringEngine Class
# ----------------------


class ClusteringEngine:
    def __init__(
        self,
        min_clusters: int = 3,
        max_clusters: int = 25,
        trials_per_batch: int = 10,
        random_state: int = 42,
        embedding_col_name: str = "embedding_vector",
        min_noise_ratio: float = 0.03,
        max_noise_ratio: float = 0.35,
        optuna_jobs: int = -1,
        min_pareto_solutions: int = 5,
        max_trials: int = 100,
        umap_config: UMAPConfig = UMAPConfig(),
        hdbscan_config: HDBSCANConfig = HDBSCANConfig(),
        branch_config: BranchDetectionConfig = BranchDetectionConfig(),
        pca_config: PCAConfig = PCAConfig(),
    ):
        """
        Initialize the optimizer with UMAP, HDBSCAN, and Branch Detection hyperparameter settings.

        Args:
            min_clusters (int): Minimum acceptable number of clusters.
            max_clusters (int): Maximum acceptable number of clusters.
            trials_per_batch (int): Number of optimization trials for hyperparameter tuning per batch.
            random_state (int): Seed for UMAP, PCA, and Optuna to ensure reproducibility.
            embedding_col_name (str): Name of the column containing embedding vectors.
            min_noise_ratio (float): Minimum acceptable noise ratio.
            max_noise_ratio (float): Maximum acceptable noise ratio.
            optuna_jobs (int): Number of parallel jobs to run during Optuna optimization.
            min_pareto_solutions (int): Minimum number of Pareto-optimal solutions to obtain before stopping optimization.
            max_trials (int): Maximum number of trials to run during hyperparameter tuning to avoid infinite loops.
            umap_config (UMAPConfig): Configuration for UMAP reduction.
            hdbscan_config (HDBSCANConfig): Configuration for HDBSCAN clustering.
            branch_config (BranchDetectionConfig): Configuration for branch detection.
            pca_config (PCAConfig): Configuration for PCA preprocessing.
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.trials_per_batch = trials_per_batch
        self.random_state = random_state
        self.embedding_col_name = embedding_col_name
        self.min_noise_ratio = min_noise_ratio
        self.max_noise_ratio = max_noise_ratio
        self.optuna_jobs = optuna_jobs
        self.min_pareto_solutions = min_pareto_solutions
        self.max_trials = max_trials
        self.umap_config = umap_config
        self.hdbscan_config = hdbscan_config
        self.branch_config = branch_config
        self.pca_config = pca_config

        # ----------------------
        # Logger Initialization: Create and configure a logger for the ClusteringEngine instance.
        # ----------------------
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_models(self, trial, num_data_pts):
        """
        Create UMAP and HDBSCAN models with hyperparameters suggested by the Optuna trial.

        For UMAP, if self.umap_config.dims is provided (not None), the number of components is fixed to that value.
        Otherwise, the number of components is sampled from the range [umap_config.min_dims, umap_config.max_dims].
        Other UMAP hyperparameters (n_neighbors, min_dist, spread, learning_rate) are also sampled within their specified ranges.
        For HDBSCAN, hyperparameters are suggested based on the number of data points, with the min_cluster_size
        computed as a multiplier of the data count.

        Args:
            trial (optuna.trial.Trial): Optuna trial object to sample hyperparameters.
            num_data_pts (int): Number of data points in the dataset.

        Returns:
            tuple: A tuple containing:
                - umap.UMAP: UMAP model instance with suggested parameters.
                - hdbscan.HDBSCAN: HDBSCAN model instance with suggested parameters.
                - dict: Dictionary of UMAP parameters.
                - dict: Dictionary of HDBSCAN parameters.
        """
        # ----------------------
        # UMAP Parameter Selection:
        # ----------------------
        if self.umap_config.dims is None:
            umap_n_components = trial.suggest_int(
                "umap_n_components",
                self.umap_config.min_dims,
                self.umap_config.max_dims,
            )
        else:
            umap_n_components = self.umap_config.dims

        umap_params = {
            "n_neighbors": trial.suggest_int(
                "umap_n_neighbors",
                self.umap_config.n_neighbors_min,
                self.umap_config.n_neighbors_max,
            ),
            "min_dist": trial.suggest_float(
                "umap_min_dist",
                self.umap_config.min_dist_min,
                self.umap_config.min_dist_max,
            ),
            "spread": trial.suggest_float(
                "umap_spread", self.umap_config.spread_min, self.umap_config.spread_max
            ),
            "metric": self.umap_config.metric,
            "random_state": self.random_state,
            "learning_rate": trial.suggest_float(
                "umap_learning_rate",
                self.umap_config.learning_rate_min,
                self.umap_config.learning_rate_max,
            ),
            "init": "spectral",
            "n_components": umap_n_components,
        }
        self.logger.debug("UMAP parameters: %s", umap_params)

        # ----------------------
        # HDBSCAN Parameter Selection:
        # ----------------------
        hdbscan_params = {
            "min_cluster_size": trial.suggest_int(
                "hdbscan_min_cluster_size",
                math.ceil(
                    self.hdbscan_config.min_cluster_size_multiplier_min * num_data_pts
                ),
                math.ceil(
                    self.hdbscan_config.min_cluster_size_multiplier_max * num_data_pts
                ),
            ),
            "min_samples": trial.suggest_int(
                "hdbscan_min_samples",
                self.hdbscan_config.min_samples_min,
                self.hdbscan_config.min_samples_max,
            ),
            "cluster_selection_epsilon": trial.suggest_float(
                "hdbscan_epsilon",
                self.hdbscan_config.epsilon_min,
                self.hdbscan_config.epsilon_max,
            ),
            "metric": self.hdbscan_config.metric,
            "cluster_selection_method": self.hdbscan_config.cluster_selection_method,
            "prediction_data": True,
        }
        if self.branch_config.enabled:
            hdbscan_params["branch_detection_data"] = True
            self.logger.debug(
                "Branch detection enabled: adding branch_detection_data=True to HDBSCAN parameters."
            )
        self.logger.debug("HDBSCAN parameters: %s", hdbscan_params)

        # ----------------------
        # Return constructed models and parameter dictionaries.
        # ----------------------
        return (
            umap.UMAP(**umap_params),
            hdbscan.HDBSCAN(**hdbscan_params),
            umap_params,
            hdbscan_params,
        )

    def _default_models(self, num_data_pts):
        """
        Create UMAP and HDBSCAN models using a set of predefined default hyperparameters.
        This fallback is used when optimization fails to find any valid Pareto-optimal trials.
        Note: In the default HDBSCAN parameters, branch detection data is always enabled.

        Args:
            num_data_pts (int): Number of data points in the dataset.

        Returns:
            tuple: A tuple containing:
                - umap.UMAP: UMAP model instance with default parameters.
                - hdbscan.HDBSCAN: HDBSCAN model instance with default parameters.
                - dict: Dictionary of default UMAP parameters.
                - dict: Dictionary of default HDBSCAN parameters.
        """
        # ----------------------
        # Determine default n_components for UMAP.
        # ----------------------
        umap_n_components = (
            self.umap_config.dims if self.umap_config.dims is not None else 3
        )

        # ----------------------
        # Compute default UMAP parameters as the average of the min and max values.
        # ----------------------
        umap_params = {
            "n_neighbors": (
                self.umap_config.n_neighbors_min + self.umap_config.n_neighbors_max
            )
            // 2,
            "min_dist": (self.umap_config.min_dist_min + self.umap_config.min_dist_max)
            / 2,
            "spread": (self.umap_config.spread_min + self.umap_config.spread_max) / 2,
            "metric": self.umap_config.metric,
            "random_state": self.random_state,
            "learning_rate": (
                self.umap_config.learning_rate_min + self.umap_config.learning_rate_max
            )
            / 2,
            "init": "spectral",
            "n_components": umap_n_components,
        }
        # ----------------------
        # Compute default HDBSCAN parameters similarly.
        # ----------------------
        hdbscan_params = {
            "min_cluster_size": math.ceil(
                (
                    (
                        self.hdbscan_config.min_cluster_size_multiplier_min
                        + self.hdbscan_config.min_cluster_size_multiplier_max
                    )
                    / 2
                )
                * num_data_pts
            ),
            "min_samples": (
                self.hdbscan_config.min_samples_min
                + self.hdbscan_config.min_samples_max
            )
            // 2,
            "cluster_selection_epsilon": (
                self.hdbscan_config.epsilon_min + self.hdbscan_config.epsilon_max
            )
            / 2,
            "metric": self.hdbscan_config.metric,
            "cluster_selection_method": self.hdbscan_config.cluster_selection_method,
            "prediction_data": True,
            "branch_detection_data": True,
        }
        self.logger.debug("Default UMAP parameters: %s", umap_params)
        self.logger.debug("Default HDBSCAN parameters: %s", hdbscan_params)

        # ----------------------
        # Instantiate and return default models.
        # ----------------------
        umap_model = umap.UMAP(**umap_params)
        hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params)
        return umap_model, hdbscan_model, umap_params, hdbscan_params

    def _create_branch_params(self, trial, num_data_pts: int) -> dict:
        """
        Create BranchDetector hyperparameters from the trial.

        Returns:
            dict: Dictionary with branch detection parameters:
                - min_branch_size: int, sampled from branch_config.min_cluster_size_multiplier_min * num_data_pts
                  to branch_config.min_cluster_size_multiplier_max * num_data_pts.
                - branch_selection_persistence: float, sampled from branch_config.selection_persistence_min to branch_config.selection_persistence_max.
                - label_sides_as_branches: bool, sampled from [True, False].
        """
        branch_params = {
            "min_branch_size": trial.suggest_int(
                "branch_min_cluster_size",
                math.ceil(
                    self.branch_config.min_cluster_size_multiplier_min * num_data_pts
                ),
                math.ceil(
                    self.branch_config.min_cluster_size_multiplier_max * num_data_pts
                ),
            ),
            "branch_selection_persistence": trial.suggest_float(
                "branch_selection_persistence",
                self.branch_config.selection_persistence_min,
                self.branch_config.selection_persistence_max,
            ),
            "label_sides_as_branches": trial.suggest_categorical(
                "branch_label_sides_as_branches", [True, False]
            ),
        }
        self.logger.debug("BranchDetector parameters: %s", branch_params)
        return branch_params

    def _compute_metrics(self, reduced_data, labels):
        """
        Compute clustering metrics including silhouette score and negative noise ratio.

        This function calculates the silhouette score for non-noise points (where noise is denoted by label -1)
        and computes the negative noise ratio. Returns None if there's only one cluster or too few non-noise points.

        Args:
            reduced_data (np.ndarray): Data after dimensionality reduction.
            labels (np.ndarray): Cluster labels assigned by HDBSCAN (or BranchDetector), with -1 indicating noise.

        Returns:
            dict or None: Dictionary containing 'silhouette' and 'neg_noise' if valid, otherwise None.
        """
        # ----------------------
        # Identify non-noise points (labels != -1).
        # ----------------------
        mask = labels != -1
        self.logger.debug("Computed non-noise mask: %s (sum=%d)", mask, np.sum(mask))

        # ----------------------
        # Check if there are enough clusters or non-noise points.
        # ----------------------
        if len(np.unique(labels)) <= 1 or np.sum(mask) < 2:
            self.logger.debug(
                "Not enough clusters or non-noise points to compute metrics."
            )
            return None

        # ----------------------
        # Compute silhouette score for non-noise data.
        # ----------------------
        silhouette = silhouette_score(
            X=reduced_data[mask], labels=labels[mask], metric="euclidean"
        )

        # ----------------------
        # Calculate negative noise ratio.
        # ----------------------
        neg_noise = -((labels == -1).sum() / len(labels))
        self.logger.debug(
            "Computed metrics: silhouette=%.4f, neg_noise=%.4f", silhouette, neg_noise
        )
        return {"silhouette": silhouette, "neg_noise": neg_noise}

    def _triple_objective(self, trial, embeddings):
        """
        Objective function for Optuna optimization combining three metrics.

        This function applies UMAP and HDBSCAN with hyperparameters from the trial, computes clustering metrics,
        and returns a triple of objectives:
            - Silhouette score (to maximize).
            - Negative noise ratio (to maximize, which corresponds to minimizing the noise ratio).
            - Negative number of clusters (to maximize, i.e. favor fewer clusters).

        If any errors occur or constraints are not met (e.g., noise ratio or cluster count outside allowed range),
        the function returns a triple of -infinity values to indicate an invalid trial.

        Args:
            trial (optuna.trial.Trial): Optuna trial for hyperparameter suggestions.
            embeddings (np.ndarray): Array of embedding vectors.

        Returns:
            list: A list containing [silhouette, negative noise ratio, negative number of clusters].
        """
        # ----------------------
        # Start triple-objective evaluation.
        # ----------------------
        self.logger.debug(
            "Starting triple-objective evaluation for trial number: %s",
            trial.number if hasattr(trial, "number") else "N/A",
        )
        try:
            # ----------------------
            # Create models using trial-specific hyperparameters.
            # ----------------------
            umap_model, hdbscan_model, _, _ = self._create_models(
                trial, len(embeddings)
            )
            self.logger.debug(
                "Models created for trial %s",
                trial.number if hasattr(trial, "number") else "N/A",
            )

            # ----------------------
            # Apply UMAP dimensionality reduction.
            # ----------------------
            reduced_data = umap_model.fit_transform(embeddings)
            self.logger.debug(
                "UMAP reduction completed for trial %s",
                trial.number if hasattr(trial, "number") else "N/A",
            )

            # ----------------------
            # Run HDBSCAN clustering (with optional branch detection).
            # ----------------------
            hdbscan_model.fit(reduced_data)
            if self.branch_config.enabled:
                branch_params = self._create_branch_params(trial, len(embeddings))
                branch_detector = BranchDetector(
                    min_branch_size=branch_params["min_branch_size"],
                    allow_single_branch=False,
                    branch_detection_method="full",  # fixed as per instructions
                    branch_selection_method="eom",  # fixed as per instructions
                    branch_selection_persistence=branch_params[
                        "branch_selection_persistence"
                    ],
                    label_sides_as_branches=branch_params["label_sides_as_branches"],
                )
                labels = branch_detector.fit_predict(hdbscan_model)  # type: ignore
            else:
                labels = hdbscan_model.labels_

            self.logger.debug("Clustering produced labels: %s", labels)

            # ----------------------
            # Compute clustering metrics.
            # ----------------------
            metrics_result = self._compute_metrics(
                reduced_data=reduced_data, labels=labels
            )
            if metrics_result is None:
                self.logger.debug(
                    "Metrics result invalid for trial %s; returning -inf objectives.",
                    trial.number if hasattr(trial, "number") else "N/A",
                )
                return [float("-inf")] * 3

            s = metrics_result["silhouette"]
            neg_noise = metrics_result["neg_noise"]
            noise_ratio = -neg_noise  # actual noise ratio

            # ----------------------
            # Enforce noise ratio constraints.
            # ----------------------
            if noise_ratio < self.min_noise_ratio or noise_ratio > self.max_noise_ratio:
                self.logger.debug(
                    "Trial %s failed noise ratio constraints (noise_ratio=%.4f).",
                    trial.number if hasattr(trial, "number") else "N/A",
                    noise_ratio,
                )
                return [float("-inf")] * 3

            # ----------------------
            # Compute number of clusters (excluding noise).
            # ----------------------
            k = len(set(labels) - {-1})
            self.logger.debug(
                "Trial %s metrics: silhouette=%.4f, neg_noise=%.4f, clusters=%d",
                trial.number if hasattr(trial, "number") else "N/A",
                s,
                neg_noise,
                k,
            )

            # ----------------------
            # Enforce cluster count constraints.
            # ----------------------
            if k < self.min_clusters or k > self.max_clusters:
                self.logger.debug(
                    "Trial %s failed cluster count constraints (k=%d).",
                    trial.number if hasattr(trial, "number") else "N/A",
                    k,
                )
                return [float("-inf")] * 3

            neg_k = -k  # Fewer clusters is better.
            return [s, neg_noise, neg_k]
        except Exception as e:
            self.logger.error("Trial failed with error: %s", str(e))
            return [float("-inf")] * 3

    def _run_optuna(self, embeddings, num_data_pts):
        """
        Run Optuna hyperparameter optimization and return the study.
        This version includes a callback that logs a message every time a trial is completed.
        The log indicates both the total number of solutions and the number of Pareto-optimal ones.
        """
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=self.random_state),
        )
        self.logger.debug("Optuna study created with multi-objective directions.")

        # ----------------------
        # Define a callback to log trial results after each trial.
        # ----------------------
        def logging_callback(study, trial):
            total_solutions = len(study.trials)
            # Only count trials with valid (non -inf) objective values as valid solutions.
            pareto_trials = [
                t for t in study.best_trials if not any(math.isinf(x) for x in t.values)
            ]
            pareto_count = len(pareto_trials)
            # Check if the current trial is in the current Pareto front.
            if trial in pareto_trials:
                self.logger.info(
                    "Pareto-optimal solution found!\n%d solutions found total.\n%d pareto-optimal",
                    total_solutions,
                    pareto_count,
                )

        total_trials = 0

        # ----------------------
        # Run optimization in batches until we reach the desired number of Pareto-optimal solutions or max_trials.
        # ----------------------
        while total_trials < self.max_trials:
            remaining_trials = min(
                self.trials_per_batch, self.max_trials - total_trials
            )
            self.logger.debug(
                "Running %d additional trials. Total trials so far: %d",
                remaining_trials,
                total_trials,
            )
            study.optimize(
                lambda trial: self._triple_objective(trial, embeddings),
                n_trials=remaining_trials,
                n_jobs=self.optuna_jobs,
                show_progress_bar=True,
                callbacks=[logging_callback],
            )
            total_trials = len(study.trials)
            self.logger.debug("Total trials completed: %d", total_trials)
            pareto_trials = [
                t for t in study.best_trials if not any(math.isinf(x) for x in t.values)
            ]
            self.logger.debug(
                "Current number of Pareto-optimal solutions: %d", len(pareto_trials)
            )
            if len(pareto_trials) >= self.min_pareto_solutions:
                self.logger.debug(
                    "Desired number of Pareto-optimal solutions reached: %d",
                    len(pareto_trials),
                )
                break
            else:
                self.logger.debug(
                    "Desired number of Pareto-optimal solutions not reached yet. Continuing trials."
                )
        self.logger.debug(
            "Optuna optimization completed with a total of %d trials.", total_trials
        )
        return study

    def _euclidean_distance_3d(self, x1, y1, z1, x2, y2, z2) -> float:
        """
        Compute the Euclidean distance between two points in 3D space.

        Args:
            x1, y1, z1 (float): Coordinates of the first point.
            x2, y2, z2 (float): Coordinates of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        # ----------------------
        # Calculate the Euclidean distance using the standard 3D distance formula.
        # ----------------------
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        self.logger.debug(
            "Computed Euclidean distance between (%.4f, %.4f, %.4f) and (%.4f, %.4f, %.4f): %.4f",
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            distance,
        )
        return distance

    def _get_best_solution(self, pareto_trials):
        """
        Select the best trial from the Pareto-optimal solutions using the TOPSIS method.

        The Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) is applied on the Pareto front.
        Each trial's metrics (silhouette, negative noise, and negative number of clusters) are normalized,
        and distances to the ideal and anti-ideal solutions are computed. The trial with the highest TOPSIS
        score is selected.

        Args:
            pareto_trials (list): List of Pareto-optimal trials with valid metric values.

        Returns:
            tuple: A tuple containing:
                - best_trial (optuna.trial.FrozenTrial): The selected best trial.
                - str: A string indicating the selection method ("pareto_topsis").

        Raises:
            ValueError: If no valid Pareto-optimal solutions are found.
        """
        # ----------------------
        # Log the number of Pareto-optimal trials.
        # ----------------------
        self.logger.info(
            "Selecting best solution from %d Pareto trials.", len(pareto_trials)
        )
        if pareto_trials:
            try:
                trial_details = []
                # ----------------------
                # Extract and collect metrics from each Pareto trial.
                # ----------------------
                for t in pareto_trials:
                    trial_details.append(
                        {
                            "trial": t,
                            "silhouette": t.values[0],
                            "neg_noise": t.values[1],
                            "neg_k": t.values[2],
                        }
                    )
                self.logger.info("Extracted trial details: %s", trial_details)

                # ----------------------
                # Collect lists of individual metrics for normalization.
                # ----------------------
                sil_vals = [d["silhouette"] for d in trial_details]
                noise_vals = [d["neg_noise"] for d in trial_details]
                k_vals = [d["neg_k"] for d in trial_details]

                # ----------------------
                # Define and compute normalization factors.
                # ----------------------
                def norm_factor(vals) -> float:
                    return math.sqrt(sum(v * v for v in vals))

                sil_norm = norm_factor(sil_vals)
                noise_norm = norm_factor(noise_vals)
                k_norm = norm_factor(k_vals)
                self.logger.info(
                    "Normalization factors: sil_norm=%.4f, noise_norm=%.4f, k_norm=%.4f",
                    sil_norm,
                    noise_norm,
                    k_norm,
                )

                # ----------------------
                # Normalize each trial's metrics.
                # ----------------------
                normalized = []
                for d in trial_details:
                    s_norm = d["silhouette"] / sil_norm if sil_norm != 0 else 0
                    n_norm = d["neg_noise"] / noise_norm if noise_norm != 0 else 0
                    k_norm_val = d["neg_k"] / k_norm if k_norm != 0 else 0
                    normalized.append(
                        {**d, "s_norm": s_norm, "n_norm": n_norm, "k_norm": k_norm_val}
                    )
                self.logger.info("Normalized trial metrics: %s", normalized)

                # ----------------------
                # Determine ideal and anti-ideal normalized values.
                # ----------------------
                s_norm_vals = [item["s_norm"] for item in normalized]
                n_norm_vals = [item["n_norm"] for item in normalized]
                k_norm_vals = [item["k_norm"] for item in normalized]

                ideal_s = max(s_norm_vals)
                ideal_n = max(n_norm_vals)
                ideal_k = max(k_norm_vals)

                anti_s = min(s_norm_vals)
                anti_n = min(n_norm_vals)
                anti_k = min(k_norm_vals)
                self.logger.info(
                    "Ideal values: (%.4f, %.4f, %.4f), Anti-ideal values: (%.4f, %.4f, %.4f)",
                    ideal_s,
                    ideal_n,
                    ideal_k,
                    anti_s,
                    anti_n,
                    anti_k,
                )

                # ----------------------
                # Compute TOPSIS scores for each trial.
                # ----------------------
                topsised = []
                for item in normalized:
                    dist_ideal = self._euclidean_distance_3d(
                        item["s_norm"],
                        item["n_norm"],
                        item["k_norm"],
                        ideal_s,
                        ideal_n,
                        ideal_k,
                    )
                    dist_anti = self._euclidean_distance_3d(
                        item["s_norm"],
                        item["n_norm"],
                        item["k_norm"],
                        anti_s,
                        anti_n,
                        anti_k,
                    )
                    topsis_score = (
                        dist_anti / (dist_ideal + dist_anti)
                        if (dist_ideal + dist_anti) != 0
                        else 0
                    )
                    topsised.append(
                        {
                            **item,
                            "dist_ideal": dist_ideal,
                            "dist_anti": dist_anti,
                            "score": topsis_score,
                        }
                    )

                self.logger.info("\n*** TOPSIS on Pareto front ***")
                for i, item in enumerate(
                    sorted(topsised, key=lambda x: -x["score"]), 1
                ):
                    self.logger.info(
                        "%d) Trial #%d - Score: %.4f",
                        i,
                        item["trial"].number,
                        item["score"],
                    )
                    self.logger.info("    Silhouette: %.4f", item["silhouette"])
                    self.logger.info("    -Noise:     %.4f", item["neg_noise"])
                    self.logger.info("    -k:         %.4f", item["neg_k"])

                # ----------------------
                # Select the trial with the highest TOPSIS score.
                # ----------------------
                best_sol = max(topsised, key=lambda x: x["score"])
                best_trial = best_sol["trial"]
                self.logger.info(
                    "\nSelected by TOPSIS => Trial #%d with Score = %.4f",
                    best_trial.number,
                    best_sol["score"],
                )
                self.logger.debug("Best trial selected: %s", best_trial)
                return best_trial, "pareto_topsis"
            except Exception as e:
                self.logger.error("TOPSIS failed with error: %s", str(e))
        # ----------------------
        # If no valid Pareto solutions, log error and raise exception.
        # ----------------------
        self.logger.error("No valid Pareto optimal solutions found... Raising error.")
        raise ValueError(
            "No valid solutions found during hyperparameter optimization. Try again with more data."
        )

    def _interpret_metric(self, metric_name, value):
        """
        Automatically interpret clustering metrics based on predefined ranges.

        Args:
            metric_name (str): The name of the metric to interpret.
            value (float or int): The value of the metric.

        Returns:
            str: Interpretation message for the metric.
        """
        match metric_name:
            case "n_clusters":
                match value:
                    case v if self.min_clusters <= v <= self.min_clusters * (1 + 0.15):
                        return "a bit low. Consider re-running if silhouette score is low and/or noise is high."
                    case v if self.max_clusters * (1 - 0.15) <= v <= self.max_clusters:
                        return "kinda high. Consider re-running if silhouette score is high and/or noise is low."
                    case _:
                        return "OK."
            case "noise_ratio":
                match value:
                    case v if v < self.min_noise_ratio:
                        return "too good to be true. Consider re-running, especially if number of clusters is high and/or silhouette score is low."
                    case v if self.min_noise_ratio <= v <= self.max_noise_ratio:
                        return "OK."
                    case _:
                        return "too high. Consider re-running, especially if number of clusters is low and/or avg silhouette score is high."
            case "silhouette_score":
                match value:
                    case v if 0.47 <= v < 0.77:
                        return "good."
                    case v if 0.35 <= v < 0.47:
                        return "so-so. Could be better. Consider re-running."
                    case v if v < 0.35:
                        return "poor. Consider re-running, especially if number of clusters is low and/or noise is low."
                    case _:
                        return "too good to be true. Consider re-running, especially if number of clusters is high."
            case _:
                return "unknown metric"

    def _pca_preprocess(self, df: pd.DataFrame):
        """
        Preprocess the DataFrame using PCA to reduce the dimensionality of the embedding vectors.
        This method uses a binary search to find the smallest number of PCA components such that
        the cumulative explained variance ratio is at least the threshold defined by pca_config.target_evr.
        The original embedding column (specified by self.embedding_col_name) is replaced with the PCA-reduced vectors.

        Args:
            df (pd.DataFrame): A copy of the original DataFrame containing the embedding vectors.

        Returns:
            dict: A dictionary containing:
                - 'pcd_reduced_df': The DataFrame with the embedding column replaced by PCA-reduced vectors.
                - 'pca_model': The fitted PCA model, or None if no suitable dimension was found.
        """
        # ----------------------
        # Start PCA preprocessing.
        # ----------------------
        self.logger.info(
            "Starting PCA preprocessing with binary search to achieve an explained variance ratio >= %.2f",
            self.pca_config.target_evr if self.pca_config.target_evr is not None else 0,
        )
        X = np.vstack(df[self.embedding_col_name].values)  # type: ignore
        orig_dim = X.shape[1]
        n_samples = X.shape[0]
        self.logger.debug("Original embeddings shape: %s", X.shape)

        # ----------------------
        # If data dimensionality exceeds number of samples, log a message.
        # ----------------------
        high = min(orig_dim, n_samples)
        if orig_dim > n_samples:
            self.logger.info(
                "Data dimensionality %d exceeds number of samples %d. PCA optimization will only search for a reduced dimension between 1 and %d. If you can, try running again with more data points.",
                orig_dim,
                n_samples,
                high,
            )

        # ----------------------
        # Set search bounds and initialize best_n_components.
        # ----------------------
        low = 1
        best_n_components = high

        # ----------------------
        # Binary search for minimal number of components satisfying target EVR.
        # ----------------------
        while low <= high:
            mid = (low + high) // 2
            pca = PCA(
                n_components=mid,
                random_state=self.random_state,
                svd_solver="randomized",
            )
            pca.fit(X)
            evr = float(np.sum(pca.explained_variance_ratio_))
            self.logger.debug(
                "Binary search trial: n_components=%d, EVR=%.4f", mid, evr
            )

            if (
                self.pca_config.target_evr is not None
                and evr >= self.pca_config.target_evr
            ):
                best_n_components = mid
                high = mid - 1
            else:
                low = mid + 1

        # ----------------------
        # Re-fit PCA with the best number of components and evaluate EVR.
        # ----------------------
        pca_model = PCA(
            n_components=best_n_components,
            random_state=self.random_state,
            svd_solver="randomized",
        )
        pca_model.fit(X)
        evr_final = float(np.sum(pca_model.explained_variance_ratio_))

        # ----------------------
        # If target EVR is not met and data dimensionality exceeds number of samples, return original df.
        # ----------------------
        if (
            self.pca_config.target_evr is not None
            and evr_final < self.pca_config.target_evr
            and n_samples < orig_dim
        ):
            self.logger.info(
                "No suitable dimension found which retains %.2f of explained variance. Consider lowering the target_evr parameter.",
                self.pca_config.target_evr,
            )
            return {"pcd_reduced_df": df, "pca_model": None}

        # ----------------------
        # Transform the data using the fitted PCA model.
        # ----------------------
        X_reduced = pca_model.transform(X)
        self.logger.info(
            "PCA preprocessing complete: selected n_components=%d with EVR=%.4f",
            best_n_components,
            evr_final,
        )
        # ----------------------
        # Replace the embedding column with PCA-reduced vectors.
        # ----------------------
        df[self.embedding_col_name] = [list(row) for row in X_reduced]
        return {"pcd_reduced_df": df, "pca_model": pca_model}

    def _final_clustering(
        self, best_umap, best_hdbscan, embeddings: np.ndarray, best_trial
    ) -> tuple:
        """
        Perform final clustering using the provided UMAP and HDBSCAN models.

        Args:
            best_umap: The final UMAP model instance.
            best_hdbscan: The final HDBSCAN model instance.
            embeddings (np.ndarray): The embedding vectors.
            best_trial: The best Optuna trial (or None if fallback was used).

        Returns:
            tuple: A tuple containing:
                - final_labels: Cluster labels.
                - membership: Membership probabilities.
                - outlier_scores_final: Outlier scores.
                - core_flags: Boolean array indicating core points.
                - reduced_coords: The reduced dimension coordinates.
                - branch_detector_final: The fitted BranchDetector if branch detection is enabled, else None.
        """
        # ----------------------
        # Fit UMAP to obtain reduced dimension coordinates.
        # ----------------------
        reduced_coords = best_umap.fit_transform(embeddings)
        self.logger.debug(
            "Final UMAP reduction complete. Reduced coordinates shape: %s",
            reduced_coords.shape,
        )

        # ----------------------
        # Predict clusters using HDBSCAN (and BranchDetector if enabled).
        # ----------------------
        if self.branch_config.enabled:
            best_hdbscan.fit(reduced_coords)
            self.logger.debug(
                "HDBSCAN model fitted on reduced coordinates for branch detection."
            )
            if best_trial is not None:
                branch_min_cluster_size = best_trial.params.get(
                    "branch_min_cluster_size",
                    math.ceil(
                        self.branch_config.min_cluster_size_multiplier_min
                        * len(reduced_coords)
                    ),
                )
                branch_selection_persistence = best_trial.params.get(
                    "branch_selection_persistence",
                    self.branch_config.selection_persistence_min,
                )
                branch_label_sides = best_trial.params.get(
                    "branch_label_sides_as_branches",
                    self.branch_config.label_sides_as_branches,
                )
            else:
                branch_min_cluster_size = math.ceil(
                    self.branch_config.min_cluster_size_multiplier_min
                    * len(reduced_coords)
                )
                branch_selection_persistence = (
                    self.branch_config.selection_persistence_min
                )
                branch_label_sides = self.branch_config.label_sides_as_branches

            branch_detector_final = BranchDetector(
                min_branch_size=branch_min_cluster_size,
                allow_single_branch=False,
                branch_detection_method="full",  # fixed as per instructions
                branch_selection_method="eom",  # fixed as per instructions
                branch_selection_persistence=branch_selection_persistence,
                label_sides_as_branches=branch_label_sides,
            )
            final_labels = branch_detector_final.fit_predict(best_hdbscan)  # type: ignore
            membership = branch_detector_final.probabilities_
            self.logger.debug("BranchDetector produced branch labels.")

            # ----------------------
            # Compute core point flags using a percentile-based threshold on membership probabilities per cluster.
            # For each cluster (excluding noise), compute the threshold as the (100 - outlier_threshold)th percentile
            # of membership values, and mark points with membership greater than or equal to that threshold as core.
            # ----------------------
            core_flags = np.zeros_like(membership, dtype=bool)
            unique_labels = np.unique(final_labels)
            self.logger.debug(
                "Processing per-cluster membership thresholds for core point detection."
            )
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise points
                cluster_indices = np.where(final_labels == label)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_membership = membership[cluster_indices]
                threshold = np.percentile(
                    cluster_membership, 100 - self.hdbscan_config.outlier_threshold
                )
                self.logger.info(
                    "Cluster %s: computed membership threshold %.4f", label, threshold
                )
                core_flags[cluster_indices] = cluster_membership >= threshold
            # Explicitly mark noise points as non-core.
            noise_indices = np.where(final_labels == -1)[0]
            core_flags[noise_indices] = False
            outlier_scores_final = np.full(len(final_labels), np.nan)
        else:
            final_labels = best_hdbscan.fit_predict(reduced_coords)  # type: ignore
            membership = best_hdbscan.probabilities_
            outlier_scores_final = best_hdbscan.outlier_scores_
            self.logger.debug("HDBSCAN clustering completed without branch detection.")

            # ----------------------
            # Compute core point flags using a percentile-based threshold on outlier scores per cluster.
            # For each cluster (excluding noise), compute the threshold as the outlier_threshold percentile
            # of outlier scores, and mark points with scores lower than that threshold as core.
            # ----------------------
            core_flags = np.zeros_like(outlier_scores_final, dtype=bool)
            unique_labels = np.unique(final_labels)
            self.logger.debug(
                "Processing per-cluster outlier score thresholds for core point detection."
            )
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise cluster
                cluster_indices = np.where(final_labels == label)[0]
                if len(cluster_indices) == 0:
                    continue
                # Invert outlier scores so that higher values indicate more central points
                centrality = -outlier_scores_final[cluster_indices]
                threshold = np.percentile(
                    centrality, 100 - self.hdbscan_config.outlier_threshold
                )
                self.logger.info(
                    "Cluster %s: computed centrality threshold %.4f", label, threshold
                )
                core_flags[cluster_indices] = centrality >= threshold

            # Explicitly mark noise points as non-core.
            noise_indices = np.where(final_labels == -1)[0]
            core_flags[noise_indices] = False
            branch_detector_final = None

        # Check for clusters without core points and raise a warning.
        for label in np.unique(final_labels):
            if label == -1:
                continue  # Skip noise cluster
            cluster_indices = np.where(final_labels == label)[0]
            if not core_flags[cluster_indices].any():
                self.logger.warning(
                    "WARNING: outlier_threshold of %s resulted in Cluster %s having no core points after clustering. Labeling will FAIL if you try to label this dataframe. Consider re-running with higher outlier_threshold.",
                    self.hdbscan_config.outlier_threshold,
                    label,
                )

        # ----------------------
        # Return final clustering results and BranchDetector instance.
        # ----------------------
        return (
            final_labels,
            membership,
            outlier_scores_final,
            core_flags,
            reduced_coords,
            branch_detector_final,
        )

    def _compute_overall_metrics(self, final_labels, reduced_coords, dims_final):
        """
        Compute overall clustering metrics (noise ratio, number of clusters, silhouette score).

        Args:
            final_labels (np.ndarray): Final cluster labels.
            reduced_coords (np.ndarray): The reduced dimension coordinates.
            dims_final (int): The number of UMAP dimensions used.

        Returns:
            dict: A dictionary of key clustering metrics.
        """
        # ----------------------
        # Compute noise ratio and number of clusters.
        # ----------------------
        noise_ratio = (final_labels == -1).sum() / len(final_labels)
        n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
        best_sil_score = None
        mask = final_labels != -1
        # ----------------------
        # Compute silhouette score for non-noise points if possible.
        # ----------------------
        if np.sum(mask) >= 2:
            best_sil_score = silhouette_score(
                X=reduced_coords[mask], labels=final_labels[mask], metric="euclidean"
            )
            self.logger.debug(
                "Computed silhouette score for non-noise points: %.4f", best_sil_score
            )
        metrics_dict = {
            "reduced_dimensions": dims_final,
            "n_clusters": n_clusters,
            "noise_ratio": round(noise_ratio, 2),
        }
        if best_sil_score is not None:
            metrics_dict["silhouette_score"] = round(best_sil_score, 2)
        return metrics_dict

    def _log_cluster_sizes(self, final_labels):
        """
        Log the sizes of the clusters.
        """
        # ----------------------
        # Determine unique labels and log sizes.
        # ----------------------
        unique_labels = sorted(set(final_labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.logger.info("\nCluster sizes:")
        for label in unique_labels:
            size = (final_labels == label).sum()
            self.logger.info(
                "  Cluster %d: %d points (%.1f%%)",
                label,
                size,
                size / len(final_labels) * 100,
            )
            self.logger.debug("Cluster %d size: %d", label, size)

    def optimize(self, filtered_df: pd.DataFrame):
        """
        Optimize UMAP and HDBSCAN hyperparameters and perform clustering on the input DataFrame.

        This method takes a DataFrame containing an embedding vector column (specified by self.embedding_col_name),
        first applies PCA preprocessing to reduce the dimensionality of the embeddings (ensuring that the cumulative
        explained variance ratio is at least pca_config.target_evr), then performs hyperparameter tuning using a triple-objective
        optimization (silhouette score, negative noise ratio, and negative cluster count), and selects the best model using TOPSIS
        on the Pareto front. If no valid Pareto-optimal trial is found, the method falls back to default hyperparameters.

        The final DataFrame is augmented with:
            - A 'reduced_vector' column containing the reduced-dimension coordinates as a list for each entry.
            - Cluster labels in 'cluster_id'.
            - Membership strengths, outlier scores, and a 'core_point' boolean column.

        Args:
            filtered_df (pd.DataFrame): Input DataFrame with an embedding column specified by self.embedding_col_name.

        Returns:
            dict: A dictionary containing:
                - 'clustered_df': The DataFrame with added clustering results.
                - 'umap_model': The final UMAP model instance.
                - 'hdbscan_model': The final HDBSCAN model instance.
                - 'pca_model': The PCA model instance used for preprocessing.
                - 'metrics_dict': Dictionary of key clustering metrics.
                - 'branch_detector': The fitted BranchDetector if branch detection is enabled, else None.
        """
        branch_detector_final = None  # Will hold the fitted BranchDetector if used.
        try:
            # ----------------------
            # Validate input DataFrame.
            # ----------------------
            self.logger.debug(
                "Starting optimization process on DataFrame with %d rows",
                len(filtered_df),
            )
            if self.embedding_col_name not in filtered_df.columns:
                self.logger.error(
                    "Input DataFrame must contain a(n) %s column.",
                    self.embedding_col_name,
                )
                raise ValueError(
                    f"Missing {self.embedding_col_name} column in input DataFrame."
                )

            # ----------------------
            # PCA Preprocessing: Reduce dimensionality if target EVR is provided.
            # ----------------------
            if self.pca_config.target_evr is not None:
                pca_result = self._pca_preprocess(filtered_df.copy())
                filtered_df = pca_result["pcd_reduced_df"]
                pca_model = pca_result["pca_model"]
                self.logger.debug(
                    "PCA preprocessing complete. Updated DataFrame with PCA-reduced embeddings."
                )
            else:
                pca_model = None
                self.logger.debug(
                    "target_pca_evr was set to None. Skipping PCA preprocessing step."
                )

            # ----------------------
            # Convert embedding vectors to a 2D numpy array.
            # ----------------------
            embeddings = np.vstack(filtered_df[self.embedding_col_name].values)  # type: ignore
            num_data_pts = len(filtered_df)
            self.logger.debug(
                "Converted embedding vectors to numpy array with shape: %s",
                embeddings.shape,
            )

            self.logger.info(
                "Starting triple-objective optimization (silhouette, -noise, -k)."
            )
            # ----------------------
            # Run Optuna optimization.
            # ----------------------
            study = self._run_optuna(embeddings, num_data_pts)
            # ----------------------
            # Extract Pareto-optimal trials.
            # ----------------------
            pareto_trials = [
                t for t in study.best_trials if not any(math.isinf(x) for x in t.values)
            ]
            self.logger.info(
                "Number of Pareto-optimal solutions: %d", len(pareto_trials)
            )

            # ----------------------
            # Log details for each Pareto trial.
            # ----------------------
            for i, trial in enumerate(pareto_trials, 1):
                s_val, neg_noise_val, neg_k_val = trial.values
                self.logger.info("\nSolution %d:", i)
                self.logger.info("    - clusters: %d", int(-neg_k_val))
                self.logger.info("    - silhouette: %.3f", s_val)
                self.logger.info("    - noise ratio: %.3f", -neg_noise_val)
                self.logger.debug(
                    "Trial #%d details: values=%s, params=%s",
                    trial.number,
                    trial.values,
                    trial.params,
                )

            # ----------------------
            # Select the best trial or fall back to default models.
            # ----------------------
            if not pareto_trials:
                self.logger.warning(
                    "No valid Pareto-optimal solutions found; falling back to default hyperparameters."
                )
                best_umap, best_hdbscan, umap_params, hdbscan_params = (
                    self._default_models(num_data_pts)
                )
                dims_final = umap_params["n_components"]
                best_trial = None
            else:
                best_trial, method_used = self._get_best_solution(pareto_trials)
                self.logger.info("Solution selection method: %s", method_used)
                s_val, neg_noise_val, neg_k_val = best_trial.values
                self.logger.info("\n*** Final Chosen Trial ***")
                self.logger.info(" - Silhouette: %.4f", s_val)
                self.logger.info(" - Neg noise:  %.4f", neg_noise_val)
                self.logger.info(" - Neg k:      %.4f", neg_k_val)
                self.logger.debug("Best trial parameters: %s", best_trial.params)
                dims_final = (
                    self.umap_config.dims
                    if self.umap_config.dims is not None
                    else best_trial.params["umap_n_components"]
                )
                best_umap, best_hdbscan, umap_params, hdbscan_params = (
                    self._create_models(best_trial, num_data_pts)
                )

            self.logger.debug("Using final UMAP dims=%d", dims_final)
            # ----------------------
            # Perform final clustering.
            # ----------------------
            (
                final_labels,
                membership,
                outlier_scores_final,
                core_flags,
                reduced_coords,
                branch_detector_final,
            ) = self._final_clustering(best_umap, best_hdbscan, embeddings, best_trial)

            # ----------------------
            # Validate that membership probabilities match DataFrame length.
            # ----------------------
            if len(membership) != len(filtered_df):
                raise AssertionError(
                    "Mismatch between probabilities and dataframe length."
                )

            # ----------------------
            # Append clustering results to the DataFrame.
            # ----------------------
            filtered_df["membership_strength"] = membership
            filtered_df["core_point"] = core_flags
            filtered_df["outlier_score"] = outlier_scores_final
            filtered_df["reduced_vector"] = [list(row) for row in reduced_coords]
            filtered_df["cluster_id"] = final_labels

            # ----------------------
            # Compute overall clustering metrics.
            # ----------------------
            noise_ratio = (final_labels == -1).sum() / len(final_labels)
            n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)

            best_sil_score = None
            mask = final_labels != -1
            if np.sum(mask) >= 2:
                best_sil_score = silhouette_score(
                    X=reduced_coords[mask],
                    labels=final_labels[mask],
                    metric="euclidean",
                )
                self.logger.debug(
                    "Computed silhouette score for non-noise points: %.4f",
                    best_sil_score,
                )

            self.logger.info("\n*** Final Clustering Results ***:")
            self.logger.info("Dimensionality: %d", dims_final)
            self.logger.info("Number of clusters: %d", n_clusters)
            if best_sil_score is not None:
                self.logger.info("Silhouette score: %.3f", best_sil_score)
            self.logger.info("Noise ratio: %.1f%%", noise_ratio * 100)

            # ----------------------
            # Log individual cluster sizes.
            # ----------------------
            self._log_cluster_sizes(final_labels)
            # ----------------------
            # Build metrics dictionary.
            # ----------------------
            metrics_dict = self._compute_overall_metrics(
                final_labels, reduced_coords, dims_final
            )

            # ----------------------
            # Interpret metrics for the user.
            # ----------------------
            n_clusters_result = self._interpret_metric(
                "n_clusters", metrics_dict["n_clusters"]
            )
            noise_ratio_result = self._interpret_metric(
                "noise_ratio", metrics_dict["noise_ratio"]
            )
            silhouette_result = self._interpret_metric(
                "silhouette_score", metrics_dict.get("silhouette_score", 0)
            )

            self.logger.info(
                "\n*** Metrics Interpretation *** \n"
                "----------------------------------------------------------------\n"
                "The run resulted in %d clusters, which is %s\n"
                "The run's noise_ratio of %.2f is %s\n"
                "The run's silhouette_score of %s is %s\n",
                metrics_dict["n_clusters"],
                n_clusters_result,
                metrics_dict["noise_ratio"],
                noise_ratio_result,
                metrics_dict.get("silhouette_score", "N/A"),
                silhouette_result,
            )

            self.logger.debug("Optimization process complete. Returning final results.")
            return {
                "clustered_df": filtered_df,
                "umap_model": best_umap,
                "hdbscan_model": best_hdbscan,
                "pca_model": pca_model,
                "metrics_dict": metrics_dict,
                "branch_detector": branch_detector_final,
            }
        except Exception as e:
            self.logger.error(
                "An error occurred during clustering optimization: %s", str(e)
            )
            raise


def validate_user_params(
    filtered_df: pd.DataFrame,
    # Clustering engine parameters:
    min_clusters,
    max_clusters,
    trials_per_batch,
    min_pareto_solutions,
    max_trials,
    random_state,
    embedding_col_name,
    min_noise_ratio,
    max_noise_ratio,
    optuna_jobs,
    # UMAP configuration parameters:
    umap_n_neighbors_min,
    umap_n_neighbors_max,
    umap_min_dist_min,
    umap_min_dist_max,
    umap_spread_min,
    umap_spread_max,
    umap_learning_rate_min,
    umap_learning_rate_max,
    umap_min_dims,
    umap_max_dims,
    umap_metric,
    dims,
    # HDBSCAN configuration parameters:
    hdbscan_min_cluster_size_multiplier_min,
    hdbscan_min_cluster_size_multiplier_max,
    hdbscan_min_samples_min,
    hdbscan_min_samples_max,
    hdbscan_epsilon_min,
    hdbscan_epsilon_max,
    hdbscan_metric,
    hdbscan_cluster_selection_method,
    hdbscan_outlier_threshold,
    # PCA configuration:
    target_pca_evr,
    # Branch detection configuration:
    hdbscan_branch_detection,
    branch_min_cluster_size_multiplier_min,
    branch_min_cluster_size_multiplier_max,
    branch_selection_persistence_min,
    branch_selection_persistence_max,
    branch_label_sides_as_branches,
):
    """
    Validate that all provided user parameters are of the expected types and within
    acceptable ranges. Raises a ValueError with an informative message if any
    parameter is invalid.

    This version additionally checks that:
      - filtered_df is a non-empty pandas DataFrame.
      - The required embedding column exists: if the user provides an embedding_col_name,
        it must exist in filtered_df; otherwise, the column 'embedding_vector' must exist.
    """

    # Validate filtered_df
    if not isinstance(filtered_df, pd.DataFrame):
        raise ValueError("filtered_df must be a pandas DataFrame.")
    if filtered_df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Validate embedding column existence
    if embedding_col_name is not None:
        if not isinstance(embedding_col_name, str) or not embedding_col_name:
            raise ValueError(
                "embedding_col_name must be a non-empty string if provided."
            )
        if embedding_col_name not in filtered_df.columns:
            raise ValueError(
                f"Input DataFrame must contain a column named '{embedding_col_name}'."
            )
    else:
        if "embedding_vector" not in filtered_df.columns:
            raise ValueError(
                "Input DataFrame must contain a column named 'embedding_vector' if embedding_col_name is None."
            )

    # Validate clustering engine parameters:
    if not isinstance(min_clusters, int) or min_clusters < 1:
        raise ValueError("min_clusters must be an integer greater than or equal to 1")
    if not isinstance(max_clusters, int) or max_clusters < min_clusters:
        raise ValueError(
            "max_clusters must be an integer greater than or equal to min_clusters"
        )
    if not isinstance(trials_per_batch, int) or trials_per_batch < 1:
        raise ValueError("trials_per_batch must be a positive integer")
    if not isinstance(min_pareto_solutions, int) or min_pareto_solutions < 1:
        raise ValueError("min_pareto_solutions must be a positive integer")
    if not isinstance(max_trials, int) or max_trials < 1:
        raise ValueError("max_trials must be a positive integer")
    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")
    if not isinstance(embedding_col_name, str) or not embedding_col_name:
        raise ValueError("embedding_col_name must be a non-empty string")

    # Validate noise ratio parameters (assuming values between 0 and 1 are acceptable):
    if not isinstance(min_noise_ratio, (int, float)) or not (0 <= min_noise_ratio <= 1):
        raise ValueError("min_noise_ratio must be a number between 0 and 1")
    if not isinstance(max_noise_ratio, (int, float)) or not (0 <= max_noise_ratio <= 1):
        raise ValueError("max_noise_ratio must be a number between 0 and 1")
    if min_noise_ratio > max_noise_ratio:
        raise ValueError(
            "min_noise_ratio must be less than or equal to max_noise_ratio"
        )
    if not isinstance(optuna_jobs, int):
        raise ValueError(
            "optuna_jobs must be an integer (e.g., -1 for all processors or a positive value)"
        )

    # Validate UMAP parameters:
    if not isinstance(umap_n_neighbors_min, int) or umap_n_neighbors_min < 1:
        raise ValueError(
            "umap_n_neighbors_min must be an integer greater than or equal to 1"
        )
    if (
        not isinstance(umap_n_neighbors_max, int)
        or umap_n_neighbors_max < umap_n_neighbors_min
    ):
        raise ValueError(
            "umap_n_neighbors_max must be an integer greater than or equal to umap_n_neighbors_min"
        )
    if not isinstance(umap_min_dist_min, (int, float)) or umap_min_dist_min < 0:
        raise ValueError("umap_min_dist_min must be a non-negative number")
    if (
        not isinstance(umap_min_dist_max, (int, float))
        or umap_min_dist_max < umap_min_dist_min
    ):
        raise ValueError(
            "umap_min_dist_max must be a number greater than or equal to umap_min_dist_min"
        )
    if not isinstance(umap_spread_min, (int, float)) or umap_spread_min <= 0:
        raise ValueError("umap_spread_min must be a positive number")
    if (
        not isinstance(umap_spread_max, (int, float))
        or umap_spread_max < umap_spread_min
    ):
        raise ValueError(
            "umap_spread_max must be a number greater than or equal to umap_spread_min"
        )
    if (
        not isinstance(umap_learning_rate_min, (int, float))
        or umap_learning_rate_min <= 0
    ):
        raise ValueError("umap_learning_rate_min must be a positive number")
    if (
        not isinstance(umap_learning_rate_max, (int, float))
        or umap_learning_rate_max < umap_learning_rate_min
    ):
        raise ValueError(
            "umap_learning_rate_max must be a number greater than or equal to umap_learning_rate_min"
        )
    if not isinstance(umap_min_dims, int) or umap_min_dims < 2:
        raise ValueError("umap_min_dims must be an integer greater than or equal to 2")
    if not isinstance(umap_max_dims, int) or umap_max_dims < umap_min_dims:
        raise ValueError(
            "umap_max_dims must be an integer greater than or equal to umap_min_dims"
        )
    if not isinstance(umap_metric, str) or not umap_metric:
        raise ValueError("umap_metric must be a non-empty string")
    if not isinstance(dims, int) or dims < 1:
        if dims is not None:
            raise ValueError("dims must be a positive integer")

    # Validate HDBSCAN parameters:
    if (
        not isinstance(hdbscan_min_cluster_size_multiplier_min, (int, float))
        or hdbscan_min_cluster_size_multiplier_min <= 0
    ):
        raise ValueError(
            "hdbscan_min_cluster_size_multiplier_min must be a positive number"
        )
    if (
        not isinstance(hdbscan_min_cluster_size_multiplier_max, (int, float))
        or hdbscan_min_cluster_size_multiplier_max
        < hdbscan_min_cluster_size_multiplier_min
    ):
        raise ValueError(
            "hdbscan_min_cluster_size_multiplier_max must be a number greater than or equal to hdbscan_min_cluster_size_multiplier_min"
        )
    if not isinstance(hdbscan_min_samples_min, int) or hdbscan_min_samples_min < 1:
        raise ValueError(
            "hdbscan_min_samples_min must be an integer greater than or equal to 1"
        )
    if (
        not isinstance(hdbscan_min_samples_max, int)
        or hdbscan_min_samples_max < hdbscan_min_samples_min
    ):
        raise ValueError(
            "hdbscan_min_samples_max must be an integer greater than or equal to hdbscan_min_samples_min"
        )
    if not isinstance(hdbscan_epsilon_min, (int, float)) or hdbscan_epsilon_min < 0:
        raise ValueError("hdbscan_epsilon_min must be a non-negative number")
    if (
        not isinstance(hdbscan_epsilon_max, (int, float))
        or hdbscan_epsilon_max < hdbscan_epsilon_min
    ):
        raise ValueError(
            "hdbscan_epsilon_max must be a number greater than or equal to hdbscan_epsilon_min"
        )
    if not isinstance(hdbscan_metric, str) or not hdbscan_metric:
        raise ValueError("hdbscan_metric must be a non-empty string")
    if (
        not isinstance(hdbscan_cluster_selection_method, str)
        or not hdbscan_cluster_selection_method
    ):
        raise ValueError("hdbscan_cluster_selection_method must be a non-empty string")
    if not isinstance(hdbscan_outlier_threshold, int) or hdbscan_outlier_threshold < 0:
        raise ValueError("hdbscan_outlier_threshold must be a non-negative integer")

    # Validate PCA parameter:
    if not isinstance(target_pca_evr, (int, float)) or not (0 < target_pca_evr <= 1):
        raise ValueError(
            "target_pca_evr must be a number between 0 (exclusive) and 1 (inclusive)"
        )

    # Validate branch detection parameters:
    if not isinstance(hdbscan_branch_detection, bool):
        raise ValueError("hdbscan_branch_detection must be a boolean")
    if (
        not isinstance(branch_min_cluster_size_multiplier_min, (int, float))
        or branch_min_cluster_size_multiplier_min <= 0
    ):
        raise ValueError(
            "branch_min_cluster_size_multiplier_min must be a positive number"
        )
    if (
        not isinstance(branch_min_cluster_size_multiplier_max, (int, float))
        or branch_min_cluster_size_multiplier_max
        < branch_min_cluster_size_multiplier_min
    ):
        raise ValueError(
            "branch_min_cluster_size_multiplier_max must be a number greater than or equal to branch_min_cluster_size_multiplier_min"
        )
    if (
        not isinstance(branch_selection_persistence_min, (int, float))
        or branch_selection_persistence_min < 0
    ):
        raise ValueError(
            "branch_selection_persistence_min must be a non-negative number"
        )
    if (
        not isinstance(branch_selection_persistence_max, (int, float))
        or branch_selection_persistence_max < branch_selection_persistence_min
    ):
        raise ValueError(
            "branch_selection_persistence_max must be a number greater than or equal to branch_selection_persistence_min"
        )
    if not isinstance(branch_label_sides_as_branches, bool):
        raise ValueError("branch_label_sides_as_branches must be a boolean")

    # If no exception was raised, all validations have passed.
    return True


# ----------------------
# run_clustering Functional Interface
# ----------------------


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
    try:
        # Validate parameters first
        validate_user_params(
            filtered_df,
            min_clusters,
            max_clusters,
            trials_per_batch,
            min_pareto_solutions,
            max_trials,
            random_state,
            embedding_col_name,
            min_noise_ratio,
            max_noise_ratio,
            optuna_jobs,
            umap_n_neighbors_min,
            umap_n_neighbors_max,
            umap_min_dist_min,
            umap_min_dist_max,
            umap_spread_min,
            umap_spread_max,
            umap_learning_rate_min,
            umap_learning_rate_max,
            umap_min_dims,
            umap_max_dims,
            umap_metric,
            dims,
            hdbscan_min_cluster_size_multiplier_min,
            hdbscan_min_cluster_size_multiplier_max,
            hdbscan_min_samples_min,
            hdbscan_min_samples_max,
            hdbscan_epsilon_min,
            hdbscan_epsilon_max,
            hdbscan_metric,
            hdbscan_cluster_selection_method,
            hdbscan_outlier_threshold,
            target_pca_evr,
            hdbscan_branch_detection,
            branch_min_cluster_size_multiplier_min,
            branch_min_cluster_size_multiplier_max,
            branch_selection_persistence_min,
            branch_selection_persistence_max,
            branch_label_sides_as_branches,
        )
        # ----------------------
        # Log function entry.
        # ----------------------
        logging.debug("run_clustering() function called.")
        # ----------------------
        # Create configuration objects.
        # ----------------------
        umap_config = UMAPConfig(
            n_neighbors_min=umap_n_neighbors_min,
            n_neighbors_max=umap_n_neighbors_max,
            min_dist_min=umap_min_dist_min,
            min_dist_max=umap_min_dist_max,
            spread_min=umap_spread_min,
            spread_max=umap_spread_max,
            learning_rate_min=umap_learning_rate_min,
            learning_rate_max=umap_learning_rate_max,
            min_dims=umap_min_dims,
            max_dims=umap_max_dims,
            metric=umap_metric,
            dims=dims,
        )
        hdbscan_config = HDBSCANConfig(
            min_cluster_size_multiplier_min=hdbscan_min_cluster_size_multiplier_min,
            min_cluster_size_multiplier_max=hdbscan_min_cluster_size_multiplier_max,
            min_samples_min=hdbscan_min_samples_min,
            min_samples_max=hdbscan_min_samples_max,
            epsilon_min=hdbscan_epsilon_min,
            epsilon_max=hdbscan_epsilon_max,
            metric=hdbscan_metric,
            cluster_selection_method=hdbscan_cluster_selection_method,
            outlier_threshold=hdbscan_outlier_threshold,
        )
        branch_config = BranchDetectionConfig(
            enabled=hdbscan_branch_detection,
            min_cluster_size_multiplier_min=branch_min_cluster_size_multiplier_min,
            min_cluster_size_multiplier_max=branch_min_cluster_size_multiplier_max,
            selection_persistence_min=branch_selection_persistence_min,
            selection_persistence_max=branch_selection_persistence_max,
            label_sides_as_branches=branch_label_sides_as_branches,
        )
        pca_config = PCAConfig(target_evr=target_pca_evr)
        # ----------------------
        # Create an instance of ClusteringEngine.
        # ----------------------
        clustering = ClusteringEngine(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            trials_per_batch=trials_per_batch,
            min_pareto_solutions=min_pareto_solutions,
            max_trials=max_trials,
            random_state=random_state,
            embedding_col_name=embedding_col_name,
            min_noise_ratio=min_noise_ratio,
            max_noise_ratio=max_noise_ratio,
            optuna_jobs=optuna_jobs,
            umap_config=umap_config,
            hdbscan_config=hdbscan_config,
            branch_config=branch_config,
            pca_config=pca_config,
        )
        logging.debug("ClusteringEngine instance created. Calling optimize().")
        # ----------------------
        # Run the clustering optimization.
        # ----------------------
        result = clustering.optimize(filtered_df=filtered_df.copy())
        logging.debug("Clustering process completed successfully.")
        return result
    except Exception as e:
        logging.error("An error occurred in the clustering process: %s", str(e))
        raise
