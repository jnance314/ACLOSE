import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score
import optuna
import math
import os

# -----
# Placeholder for global (class) variables
# -----
dims = 2
min_clusters = 2
max_clusters = 10
n_trials = 50
# -----


def get_best_solution(
    study,
    pareto_trials,
    min_clusters,
    max_clusters
    ):
    """
    Determine the best solution with fallback strategy when Pareto front is empty/invalid.

    Args:
        study: Optuna study object
        pareto_trials: List of Pareto optimal trials
        min_clusters: Minimum number of clusters allowed
        max_clusters: Maximum number of clusters allowed

    Returns:
        best_trial: The selected trial to use
        method_used: String indicating which method was used to select the trial

    Raises:
        ValueError: If no valid solutions can be found in any trials
    """
    # ------------------------------------------------------------------
    # If we have Pareto front solutions, DO TOPSIS ON *ALL* OF THEM
    # ------------------------------------------------------------------
    if pareto_trials:
        try:
            # 1) Collect details for *all* Pareto trials, not just the first one
            trial_details = []
            for t in pareto_trials:
                trial_details.append(
                    {
                        "trial": t,
                        "silhouette": t.values[0],
                        "neg_noise": t.values[1],
                        "neg_k": t.values[2],
                    }
                )

            # 2) Run TOPSIS on the entire list of Pareto solutions
            sil_vals = [d["silhouette"] for d in trial_details]
            noise_vals = [d["neg_noise"] for d in trial_details]
            k_vals = [d["neg_k"] for d in trial_details]

            def norm_factor(vals):
                return math.sqrt(sum(v * v for v in vals))

            sil_norm_factor = norm_factor(sil_vals)
            noise_norm_factor = norm_factor(noise_vals)
            k_norm_factor = norm_factor(k_vals)

            def eucl_dist_3d(x1, y1, z1, x2, y2, z2):
                return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

            # Normalize everything, then identify ideal / anti-ideal
            normalized = []
            for d in trial_details:
                s_norm = (
                    d["silhouette"] / sil_norm_factor if sil_norm_factor != 0 else 0
                )
                n_norm = (
                    d["neg_noise"] / noise_norm_factor if noise_norm_factor != 0 else 0
                )
                k_norm = d["neg_k"] / k_norm_factor if k_norm_factor != 0 else 0
                normalized.append(
                    {**d, "s_norm": s_norm, "n_norm": n_norm, "k_norm": k_norm}
                )

            s_norm_vals = [item["s_norm"] for item in normalized]
            n_norm_vals = [item["n_norm"] for item in normalized]
            k_norm_vals = [item["k_norm"] for item in normalized]

            ideal_s = max(s_norm_vals)
            ideal_n = max(n_norm_vals)
            ideal_k = max(k_norm_vals)

            anti_s = min(s_norm_vals)
            anti_n = min(n_norm_vals)
            anti_k = min(k_norm_vals)

            topsised = []
            for item in normalized:
                s_norm = item["s_norm"]
                n_norm = item["n_norm"]
                k_norm = item["k_norm"]

                dist_ideal = eucl_dist_3d(
                    s_norm, n_norm, k_norm, ideal_s, ideal_n, ideal_k
                )
                dist_anti = eucl_dist_3d(s_norm, n_norm, k_norm, anti_s, anti_n, anti_k)

                if (dist_ideal + dist_anti) == 0:
                    topsis_score = 0
                else:
                    topsis_score = dist_anti / (dist_ideal + dist_anti)

                topsised.append(
                    {
                        **item,
                        "dist_ideal": dist_ideal,
                        "dist_anti": dist_anti,
                        "score": topsis_score,
                    }
                )

            # 3) Pick the single best solution by highest TOPSIS score
            best_sol = max(topsised, key=lambda x: x["score"])
            best_trial = best_sol["trial"]

            # Print some debug info
            print("\n*** TOPSIS on Pareto front ***")
            for i, item in enumerate(sorted(topsised, key=lambda x: -x["score"]), 1):
                print(
                    f"{i}) Trial #{item['trial'].number} - Score: {item['score']:.4f}"
                )
                print(f"    Silhouette: {item['silhouette']:.4f}")
                print(f"    -Noise:     {item['neg_noise']:.4f}")
                print(f"    -k:         {item['neg_k']:.4f}")

            print(
                f"\nSelected by TOPSIS => Trial #{best_trial.number} with Score = {best_sol['score']:.4f}"
            )

            return best_trial, "pareto_topsis"  # <--- CHANGED
        except Exception as e:
            print(f"\nTOPSIS failed with error: {str(e)}")
            # Continue to fallback strategy

    print("\nAttention: No valid Pareto optimal solutions found... Raising error.\n")

    # ------------------------------------------------------------------
    # Fallback strategy if Pareto front is empty or invalid
    # * removing for now. this causes type error (inf) if too few datapoints
    # ------------------------------------------------------------------
    # valid_trials = []
    # for trial in study.trials:
    #     if trial.state != optuna.trial.TrialState.COMPLETE:
    #         continue

    #     s_val, neg_noise_val, neg_k_val = trial.values
    #     k = int(-neg_k_val)  # Convert back from negative

    #     # Check if meets basic validity criteria
    #     if (k >= min_clusters and
    #         k <= max_clusters and
    #         s_val != float('-inf') and
    #         neg_noise_val != float('-inf')):
    #         valid_trials.append(trial)

    # if valid_trials:
    #     # Score trials using a simple weighted sum
    #     scored_trials = []
    #     for trial in valid_trials:
    #         s_val, neg_noise_val, neg_k_val = trial.values
    #         # Equal weights for each objective
    #         score = (s_val + neg_noise_val + neg_k_val) / 3
    #         scored_trials.append((score, trial))

    #     best_trial = max(scored_trials, key=lambda x: x[0])[1]
    #     return best_trial, "weighted_sum"

    raise ValueError(
        "No valid solutions found during Machine Learning step. Possibly too few messages. Consider adjusting the filters and time window and try running again."
    )


def optimize_umap_clustering(filtered_df):
    """
    Optimize UMAP + HDBSCAN hyperparameters with a triple-objective approach:
        1) Silhouette Score (maximize)
        2) Negative Noise Ratio (i.e. -noise_ratio, so that less noise => bigger objective)
        3) Negative Number of Clusters (i.e. -k, so fewer clusters => bigger objective)

    Then, from the resulting Pareto front, pick the single best solution using TOPSIS.

    Args:
        filtered_df (pd.DataFrame): DataFrame containing the data to reduce.

    Returns:
        dict: {
            'clustered_df': DataFrame with reduced dimensions and cluster labels,
            'umap_model': the best UMAP model,
            'hdbscan_model': the best HDBSCAN model,
            'metrics_dict': dictionary of final clustering metrics
        }
    """
    # ------------------------------------------------------
    # Prepare embeddings
    # ------------------------------------------------------
    embeddings = np.vstack(filtered_df['embedding_vector'].values)
    num_data_pts = len(filtered_df)

    # ------------------------------------------------------
    # Helper to create UMAP + HDBSCAN models from trial
    # ------------------------------------------------------
    def create_models(trial):
        """Create both UMAP and HDBSCAN models with trial parameters"""
        umap_params = {
            "n_neighbors": trial.suggest_int("umap_n_neighbors", 2, 25),
            "min_dist": trial.suggest_float("umap_min_dist", 0.0, 0.1),
            "spread": trial.suggest_float("umap_spread", 1.0, 10.0),
            "metric": "cosine",
            "random_state": 49,
            "learning_rate": trial.suggest_float("umap_learning_rate", 0.08, 1.0),
            "init": "spectral",
            "n_components": (
                trial.suggest_int("umap_n_components", 2, 20)
                if dims is None
                else trial.suggest_int("umap_n_components", 2, 3)
            ),
        }

        hdbscan_params = {
            "min_cluster_size": trial.suggest_int(
                "hdbscan_min_cluster_size",
                math.ceil(0.005 * num_data_pts),
                math.ceil(0.025 * num_data_pts),
            ),
            "min_samples": trial.suggest_int("hdbscan_min_samples", 2, 50),
            "cluster_selection_epsilon": trial.suggest_float(
                "hdbscan_epsilon", 0.0, 1.0
            ),
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        }

        return (
            umap.UMAP(**umap_params),
            hdbscan.HDBSCAN(**hdbscan_params),
            umap_params,
            hdbscan_params,
        )

    # ------------------------------------------------------
    # Compute silhouette and negative noise ratio
    # ------------------------------------------------------
    def compute_metrics(reduced_data, labels):
        """
        Compute silhouette score and negative noise ratio.
        Return None if invalid (only 1 cluster or all noise).
        """
        metrics = {}
        mask = labels != -1

        # Return None if no valid clusters
        if len(np.unique(labels)) <= 1 or sum(mask) < 2:
            return None

        # Silhouette score: higher is better
        silhouette = silhouette_score(
            reduced_data[mask], labels[mask], metric="euclidean"
        )
        # noise ratio => negative means "less noise" is better
        neg_noise = -((labels == -1).sum() / len(labels))

        metrics["silhouette"] = silhouette
        metrics["neg_noise"] = neg_noise
        return metrics

    # ------------------------------------------------------
    # The triple-objective function
    # ------------------------------------------------------
    def triple_objective(trial):
        """Return [silhouette, -noise_ratio, -k]."""
        try:
            umap_model, hdbscan_model, _, _ = create_models(trial)
            reduced_data = umap_model.fit_transform(embeddings)
            labels = hdbscan_model.fit_predict(reduced_data)  # type: ignore

            metrics_result = compute_metrics(reduced_data, labels)
            if metrics_result is None:
                return [float("-inf")] * 3

            s = metrics_result["silhouette"]  # bigger is better
            neg_noise = metrics_result["neg_noise"]  # bigger is better
            k = len(set(labels) - {-1})

            # * pre-filtering approach (used currently)
            # We don't want solutions in our Pareto front with less than <min_clusters> clusters
            if k < min_clusters or k > max_clusters:
                return [float("-inf")] * 3
            # * ---

            neg_k = -k  # fewer clusters => bigger is better

            return [s, neg_noise, neg_k]
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            return [float("-inf")] * 3

    # ------------------------------------------------------
    # Create the study (always triple-objective)
    # ------------------------------------------------------
    print(f"Starting triple-objective optimization (silhouette, -noise, -k).")
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=49),
    )

    # ------------------------------------------------------
    # Run the optimization
    # ------------------------------------------------------
    study.optimize(
        triple_objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True
    )

    # ------------------------------------------------------
    # Retrieve the Pareto front
    # ------------------------------------------------------
    all_pareto = study.best_trials
    # Filter out any that have ∞ or −∞ in their objective values
    pareto_trials = [t for t in all_pareto if not any(math.isinf(x) for x in t.values)]
    print("\nPareto front trials:")
    print(f"Number of Pareto-optimal solutions: {len(pareto_trials)}")

    # Print details of each Pareto optimal solution
    for i, trial in enumerate(pareto_trials, 1):
        s_val, neg_noise_val, neg_k_val = trial.values
        print(f"\nSolution {i}:")
        print(f"    - clusters: {int(-neg_k_val)}")  # Convert back from negative
        print(f"    - silhouette: {s_val:.3f}")
        print(f"    - noise ratio: {-neg_noise_val:.3f}")  # Convert back from negative

    # ------------------------------------------------------
    # Get best solution (now actually uses TOPSIS on the entire Pareto front)
    # ------------------------------------------------------
    try:
        best_trial, method_used = get_best_solution(
            study, pareto_trials, min_clusters, max_clusters
        )

        print(f"\nSolution selection method: {method_used}")
        if method_used == "weighted_sum":
            print(
                "Note: Using weighted sum fallback - no valid Pareto optimal solutions found."
            )

    except ValueError as e:
        print(f"ERROR: {str(e)}")
        raise  # Re-raise the exception to handle it at a higher level if needed

    # ------------------------------------------------------
    # Fit final models with best parameters
    # ------------------------------------------------------
    s_val, neg_noise_val, neg_k_val = best_trial.values
    print("\n*** Final Chosen Trial ***")
    print(f" - Silhouette: {s_val:.4f}")
    print(f" - Neg noise:  {neg_noise_val:.4f}")
    print(f" - Neg k:      {neg_k_val:.4f}")

    dims = best_trial.params["umap_n_components"]
    best_umap, best_hdbscan, umap_params, hdbscan_params = create_models(best_trial)
    reduced_coords = best_umap.fit_transform(embeddings)
    cluster_number = best_hdbscan.fit_predict(reduced_coords)  # type: ignore

    # Add cluster membership strengths
    assert len(best_hdbscan.probabilities_) == len(
        filtered_df
    ), f"Mismatch between probabilities ({len(best_hdbscan.probabilities_)}) and dataframe length ({len(filtered_df)})"
    filtered_df["membership_strength"] = best_hdbscan.probabilities_

    # Identify core points using outlier scores (lower => more "core-like")
    outlier_scores = best_hdbscan.outlier_scores_
    threshold = np.percentile(outlier_scores, 90)
    filtered_df["core_point"] = outlier_scores < threshold
    filtered_df["outlier_score"] = outlier_scores

    # ------------------------------------------------------
    # Calculate final stats
    # ------------------------------------------------------
    noise_ratio = (cluster_number == -1).sum() / len(cluster_number)
    n_clusters = len(set(cluster_number)) - (1 if -1 in cluster_number else 0)

    # Attempt silhouette if there is at least 2 points of non-noise
    best_sil_score = None
    mask = cluster_number != -1
    if sum(mask) >= 2:
        best_sil_score = silhouette_score(
            reduced_coords[mask],  # type: ignore
            cluster_number[mask],
            metric="euclidean",
        )

    print("\n*** Final Clustering Results ***:\n")
    print(f"Dimensionality: {dims}")
    print(f"Number of clusters: {n_clusters}")
    if best_sil_score is not None:
        print(f"Silhouette score: {best_sil_score:.3f}")
    print(f"Noise ratio: {noise_ratio:.1%}")

    # Print cluster sizes
    unique_labels = sorted(set(cluster_number))
    if -1 in unique_labels:
        unique_labels.remove(-1)
    print("\nCluster sizes:")
    for label in unique_labels:
        size = (cluster_number == label).sum()
        print(
            f"  Cluster {label}: {size} points ({size / len(cluster_number) * 100:.1f}%)"
        )

    # Add reduced coordinates and cluster labels to DataFrame
    dimension_cols = [f"dim_{i + 1}" for i in range(dims)]
    for i, col in enumerate(dimension_cols):
        filtered_df[col] = reduced_coords[:, i]  # type: ignore
    filtered_df["cluster_number"] = cluster_number

    clustered_df = filtered_df

    # ------------------------------------------------------
    # Construct metrics dictionary
    # ------------------------------------------------------
    metrics_dict = {
        "reduced_dimensions": dims,
        "n_clusters": n_clusters,
        "noise_ratio": round(float(noise_ratio), 2),
    }
    if best_sil_score is not None:
        metrics_dict["silhouette_score"] = round(float(best_sil_score), 2)

    return {
        "clustered_df": clustered_df,
        "umap_model": best_umap,
        "hdbscan_model": best_hdbscan,
        "metrics_dict": metrics_dict,
    }
