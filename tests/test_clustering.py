import math
import pytest
import numpy as np
import pandas as pd
import logging
import umap
import hdbscan
import optuna

# Import the clustering package components.
from aclose.clustering import (
    ClusteringEngine,
    run_clustering,
    UMAPConfig,
    HDBSCANConfig,
    BranchDetectionConfig,
    PCAConfig,
    validate_user_params,
)

# ----------------------
# Dummy Classes for Testing
# ----------------------


class DummyTrial:
    def __init__(self):
        self.number = 1
        self.params = {}

    def suggest_int(self, name, low, high):
        value = (low + high) // 2
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high):
        value = (low + high) / 2
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value


class DummyFrozenTrial:
    def __init__(self, number, values, params):
        self.number = number
        self.values = values
        self.params = params


# ----------------------
# Pytest Fixtures
# ----------------------


@pytest.fixture
def clustering_engine():
    """
    Create a ClusteringEngine instance with updated parameters.
    The engine is configured with:
      - UMAPConfig with dims set to 2.
      - HDBSCANConfig with:
          - min_cluster_size_multiplier set so that min cluster size is > 1,
          - and min_samples range fixed to 2 to avoid overshooting the training set size.
      - Trials parameters adjusted: trials_per_batch, min_pareto_solutions, and max_trials.
    """
    umap_config = UMAPConfig(dims=2)
    # Ensure min_cluster_size >= 2 for small datasets.
    hdbscan_config = HDBSCANConfig(
        min_cluster_size_multiplier_min=0.1,
        min_cluster_size_multiplier_max=0.2,
        min_samples_min=2,
        min_samples_max=2,  # Fix min_samples to 2 for testing small datasets.
    )
    branch_config = BranchDetectionConfig(enabled=False)
    pca_config = PCAConfig(target_evr=0.9)
    engine = ClusteringEngine(
        min_clusters=2,
        max_clusters=5,
        trials_per_batch=2,
        min_pareto_solutions=1,
        max_trials=5,
        random_state=42,
        embedding_col_name="embedding_vector",
        min_noise_ratio=0.03,
        max_noise_ratio=0.35,
        optuna_jobs=1,
        umap_config=umap_config,
        hdbscan_config=hdbscan_config,
        branch_config=branch_config,
        pca_config=pca_config,
    )
    return engine


# ----------------------
# Tests for Helper Methods
# ----------------------


def test_compute_metrics_valid(clustering_engine):
    """
    Test _compute_metrics with valid inputs.
    Supplies reduced data and labels with at least two clusters and non-noise points.
    """
    reduced_data = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    labels = np.array([0, 0, 1, 1])
    metrics = clustering_engine._compute_metrics(reduced_data, labels)
    assert metrics is not None
    assert "silhouette" in metrics
    assert "neg_noise" in metrics


def test_compute_metrics_invalid(clustering_engine):
    """
    Test _compute_metrics when only one cluster exists.
    Expecting a return of None since silhouette score cannot be computed.
    """
    reduced_data = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 0])
    metrics = clustering_engine._compute_metrics(reduced_data, labels)
    assert metrics is None


def test_euclidean_distance_3d(clustering_engine):
    """
    Test the _euclidean_distance_3d method.
    Compares the computed distance between (0,0,0) and (1,1,1) with sqrt(3).
    """
    dist = clustering_engine._euclidean_distance_3d(0, 0, 0, 1, 1, 1)
    expected = math.sqrt(3)
    assert pytest.approx(dist, rel=1e-3) == expected


def test_create_models(clustering_engine):
    """
    Test _create_models to verify that it returns valid UMAP and HDBSCAN models and parameters.
    """
    dummy_trial = DummyTrial()
    num_data_pts = 100
    umap_model, hdbscan_model, umap_params, hdbscan_params = (
        clustering_engine._create_models(dummy_trial, num_data_pts)
    )
    assert isinstance(umap_model, umap.UMAP)
    assert isinstance(hdbscan_model, hdbscan.HDBSCAN)
    # Check that the suggested parameters lie within expected bounds (using defaults from the dataclasses)
    assert umap_params["n_neighbors"] >= clustering_engine.umap_config.n_neighbors_min
    assert (
        hdbscan_params["min_samples"]
        >= clustering_engine.hdbscan_config.min_samples_min
    )


def test_default_models(clustering_engine):
    """
    Test _default_models to verify that default models are created with averaged parameter values.
    """
    num_data_pts = 50
    umap_model, hdbscan_model, umap_params, hdbscan_params = (
        clustering_engine._default_models(num_data_pts)
    )
    assert isinstance(umap_model, umap.UMAP)
    assert isinstance(hdbscan_model, hdbscan.HDBSCAN)
    expected_dims = (
        clustering_engine.umap_config.dims
        if clustering_engine.umap_config.dims is not None
        else 3
    )
    assert umap_params["n_components"] == expected_dims


def test_triple_objective_exception(clustering_engine, monkeypatch):
    """
    Test _triple_objective to confirm that if _create_models raises an exception,
    the objective returns a list of negative infinity values.
    """

    def raise_exception(*args, **kwargs):
        raise Exception("Forced error")

    monkeypatch.setattr(clustering_engine, "_create_models", raise_exception)
    dummy_trial = DummyTrial()
    embeddings = np.random.rand(10, 2)
    result = clustering_engine._triple_objective(dummy_trial, embeddings)
    assert result == [float("-inf"), float("-inf"), float("-inf")]


def test_get_best_solution(clustering_engine):
    """
    Test _get_best_solution using dummy frozen trials.
    Verifies that one of the dummy trials is selected and the method returns "pareto_topsis".
    """
    trial1 = DummyFrozenTrial(1, [0.5, -0.1, -3], {"param": 1})
    trial2 = DummyFrozenTrial(2, [0.6, -0.2, -4], {"param": 2})
    pareto_trials = [trial1, trial2]
    best_trial, method = clustering_engine._get_best_solution(pareto_trials)
    assert best_trial in pareto_trials
    assert method == "pareto_topsis"


def test_interpret_metric(clustering_engine):
    """
    Test the _interpret_metric method for different metric names and values.
    """
    # For n_clusters, test with a value equal to min_clusters (should be interpreted as a bit low or OK)
    msg_nc = clustering_engine._interpret_metric(
        "n_clusters", clustering_engine.min_clusters
    )
    assert "low" in msg_nc or "OK" in msg_nc

    # For noise_ratio, test with a value below min_noise_ratio
    msg_nr = clustering_engine._interpret_metric(
        "noise_ratio", clustering_engine.min_noise_ratio - 0.01
    )
    assert "too good" in msg_nr

    # For silhouette_score, test with a low value
    msg_ss = clustering_engine._interpret_metric("silhouette_score", 0.3)
    assert "poor" in msg_ss


def test_pca_preprocess(clustering_engine):
    """
    Test the _pca_preprocess method using a dummy DataFrame with higher-dimensional embeddings.
    """
    num_samples = 15
    embeddings = [np.random.rand(10) for _ in range(num_samples)]
    df = pd.DataFrame({"embedding_vector": embeddings})
    result = clustering_engine._pca_preprocess(df.copy())
    # Check that the result contains the expected keys.
    assert "pcd_reduced_df" in result
    assert "pca_model" in result
    # Verify that the embedding column now contains a list (of PCA-reduced values)
    reduced_df = result["pcd_reduced_df"]
    assert isinstance(reduced_df["embedding_vector"].iloc[0], list)


def test_log_cluster_sizes(clustering_engine, caplog):
    """
    Test _log_cluster_sizes by capturing log output and ensuring cluster sizes are reported.
    """
    final_labels = np.array([0, 0, 1, -1, 1, 2, 2, 2])
    with caplog.at_level(logging.INFO):
        clustering_engine._log_cluster_sizes(final_labels)
    messages = [record.message for record in caplog.records]
    assert any("Cluster 0" in msg for msg in messages)
    assert any("Cluster 1" in msg for msg in messages)
    assert any("Cluster 2" in msg for msg in messages)


# ----------------------
# Tests for High-Level Methods
# ----------------------


def test_optimize_missing_column(clustering_engine):
    """
    Test that optimize raises a ValueError when the required embedding column is missing.
    """
    df = pd.DataFrame({"not_embedding": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing"):
        clustering_engine.optimize(df)


def test_optimize_success(clustering_engine):
    """
    Test optimize with a minimal valid DataFrame.
    Verifies that the returned dictionary contains the expected keys and that the
    DataFrame is augmented with clustering result columns.
    """
    num_samples = 20
    embeddings = [np.random.rand(5) for _ in range(num_samples)]
    df = pd.DataFrame(
        {
            "embedding_vector": embeddings,
            "content": [f"sample {i}" for i in range(num_samples)],
        }
    )
    result = clustering_engine.optimize(df)
    for key in ("clustered_df", "umap_model", "hdbscan_model", "metrics_dict"):
        assert key in result
    clustered_df = result["clustered_df"]
    for col in [
        "membership_strength",
        "core_point",
        "outlier_score",
        "reduced_vector",
        "cluster_id",
    ]:
        assert col in clustered_df.columns


def test_run_clustering():
    """
    Test the run_clustering functional interface.
    Creates a dummy DataFrame and verifies that the returned dictionary
    contains all expected keys.
    Overrides HDBSCAN multipliers and min_samples to ensure a valid configuration
    for a small dataset.
    """
    num_samples = 20
    embeddings = [np.random.rand(5) for _ in range(num_samples)]
    df = pd.DataFrame(
        {
            "embedding_vector": embeddings,
            "content": [f"sample {i}" for i in range(num_samples)],
        }
    )
    result = run_clustering(
        df,
        # Override HDBSCAN parameters to ensure a valid configuration.
        hdbscan_min_cluster_size_multiplier_min=0.1,
        hdbscan_min_cluster_size_multiplier_max=0.2,
        hdbscan_min_samples_min=2,
        hdbscan_min_samples_max=2,
    )
    for key in (
        "clustered_df",
        "umap_model",
        "hdbscan_model",
        "pca_model",
        "metrics_dict",
        "branch_detector",
    ):
        assert key in result


def test_validate_user_params_valid():
    """
    Test validate_user_params with valid parameters.
    """
    df = pd.DataFrame({"embedding_vector": [np.random.rand(5) for _ in range(10)]})
    valid = validate_user_params(
        df,
        3,  # min_clusters
        10,  # max_clusters
        5,  # trials_per_batch
        2,  # min_pareto_solutions
        20,  # max_trials
        42,  # random_state
        "embedding_vector",
        0.03,
        0.35,
        -1,  # optuna_jobs (-1 for using all processors is allowed)
        2,
        25,
        0.0,
        0.1,
        1.0,
        10.0,
        0.08,
        1.0,
        2,
        20,
        "cosine",
        3,
        0.005,
        0.025,
        2,
        50,
        0.0,
        1.0,
        "euclidean",
        "eom",
        10,
        0.9,
        False,
        0.005,
        0.025,
        0.0,
        0.1,
        False,
    )
    assert valid is True


def test_validate_user_params_invalid_embedding():
    """
    Test validate_user_params to ensure it raises an error if the embedding column is missing.
    """
    df = pd.DataFrame({"wrong_column": [1, 2, 3]})
    with pytest.raises(ValueError, match="embedding"):
        validate_user_params(
            df,
            3,
            10,
            5,
            2,
            20,
            42,
            "embedding_vector",
            0.03,
            0.35,
            -1,
            2,
            25,
            0.0,
            0.1,
            1.0,
            10.0,
            0.08,
            1.0,
            2,
            20,
            "cosine",
            3,
            0.005,
            0.025,
            2,
            50,
            0.0,
            1.0,
            "euclidean",
            "eom",
            10,
            0.9,
            False,
            0.005,
            0.025,
            0.0,
            0.1,
            False,
        )
