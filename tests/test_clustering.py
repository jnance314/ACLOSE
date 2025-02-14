import math
import pytest
import numpy as np
import pandas as pd
import logging
import umap
import hdbscan
import optuna
import os

from atmose.clustering import ClusteringEngine, run_clustering

# A dummy trial to simulate Optuna suggestions.
class DummyTrial:
    def __init__(self):
        self.number = 1
        self.params = {}
    def suggest_int(self, name, low, high):
        return (low + high) // 2
    def suggest_float(self, name, low, high):
        return (low + high) / 2

# A dummy frozen trial to simulate an Optuna trial on the Pareto front.
class DummyFrozenTrial:
    def __init__(self, number, values, params):
        self.number = number
        self.values = values
        self.params = params

# A dummy study that holds best_trials.
class DummyStudy:
    def __init__(self, best_trials):
        self.best_trials = best_trials

@pytest.fixture
def clustering_engine():
    """
    Fixture to create a ClusteringEngine instance with preset parameters.
    
    The engine is configured with:
      - dims=2, min_clusters=2, max_clusters=5, n_trials=1, and a fixed random_state.
    
    This instance is used for testing various helper methods within ClusteringEngine.
    """
    engine = ClusteringEngine(
        dims=2,
        min_clusters=2,
        max_clusters=5,
        n_trials=1,
        random_state=42
    )
    return engine

def test_compute_metrics_valid(clustering_engine):
    """
    Test _compute_metrics with valid inputs.
    
    The test supplies reduced data and labels that contain at least two clusters and at least
    two non-noise points, and verifies that the returned dictionary contains both 'silhouette'
    and 'neg_noise' keys.
    """
    reduced_data = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    labels = np.array([0, 0, 1, 1])
    metrics = clustering_engine._compute_metrics(reduced_data, labels)
    assert metrics is not None
    assert "silhouette" in metrics
    assert "neg_noise" in metrics

def test_compute_metrics_invalid(clustering_engine):
    """
    Test _compute_metrics when the input data corresponds to a single cluster.
    
    Since silhouette score cannot be computed for one cluster, the method should return None.
    """
    reduced_data = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 0])
    metrics = clustering_engine._compute_metrics(reduced_data, labels)
    assert metrics is None

def test_euclidean_distance_3d(clustering_engine):
    """
    Test the _euclidean_distance_3d method.
    
    Computes the Euclidean distance between the points (0,0,0) and (1,1,1) and compares
    it to the expected value of sqrt(3).
    """
    dist = clustering_engine._euclidean_distance_3d(0, 0, 0, 1, 1, 1)
    expected = math.sqrt(3)
    assert pytest.approx(dist, rel=1e-3) == expected

def test_create_models(clustering_engine):
    """
    Test _create_models to verify that it returns valid UMAP and HDBSCAN models along with their parameters.
    
    A DummyTrial instance is used to simulate hyperparameter suggestions. The test then checks:
      - The returned UMAP model is an instance of umap.UMAP.
      - The returned HDBSCAN model is an instance of hdbscan.HDBSCAN.
      - The suggested parameters (e.g. n_neighbors, min_samples) lie within expected bounds.
    """
    dummy_trial = DummyTrial()
    num_data_pts = 100
    umap_model, hdbscan_model, umap_params, hdbscan_params = clustering_engine._create_models(dummy_trial, num_data_pts)
    assert isinstance(umap_model, umap.UMAP)
    assert isinstance(hdbscan_model, hdbscan.HDBSCAN)
    assert umap_params["n_neighbors"] >= clustering_engine.umap_n_neighbors_min
    assert hdbscan_params["min_samples"] >= clustering_engine.hdbscan_min_samples_min

def test_default_models(clustering_engine):
    """
    Test _default_models to verify that default models are created with preset average parameter values.
    
    For a given number of data points, the test ensures that:
      - The returned UMAP and HDBSCAN models are of the correct types.
      - If dims is specified, then the default UMAP n_components equals dims.
    """
    num_data_pts = 50
    umap_model, hdbscan_model, umap_params, hdbscan_params = clustering_engine._default_models(num_data_pts)
    assert isinstance(umap_model, umap.UMAP)
    assert isinstance(hdbscan_model, hdbscan.HDBSCAN)
    assert umap_params["n_components"] == clustering_engine.dims

def test_triple_objective_exception(clustering_engine, monkeypatch):
    """
    Test _triple_objective to confirm that when _create_models raises an exception,
    the objective function returns a list of negative infinity values.
    
    This simulates an error during model creation, forcing the trial to be marked invalid.
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
    
    Two dummy trials with fixed metric values are created and added to a dummy study.
    The test verifies that:
      - One of the dummy trials is selected as the best trial.
      - The selection method string returned is "pareto_topsis".
    """
    trial1 = DummyFrozenTrial(1, [0.5, -0.1, -3], {"param": 1})
    trial2 = DummyFrozenTrial(2, [0.6, -0.2, -4], {"param": 2})
    pareto_trials = [trial1, trial2]
    dummy_study = DummyStudy(pareto_trials)
    best_trial, method = clustering_engine._get_best_solution(dummy_study, pareto_trials)
    assert best_trial in pareto_trials
    assert method == "pareto_topsis"

def test_optimize_missing_column(clustering_engine):
    """
    Test the optimize method to ensure that it raises a ValueError when the required embedding column is missing.
    
    A dummy DataFrame that does not contain the 'embedding_vector' column is passed in,
    and the test expects a ValueError with a message containing "Missing".
    """
    df = pd.DataFrame({"not_embedding": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing"):
        clustering_engine.optimize(df)

def test_optimize_success(clustering_engine):
    """
    Test the optimize method with a minimal valid DataFrame.
    
    The DataFrame includes an 'embedding_vector' column (with 2D numpy arrays) and a 'content' column.
    The test sets n_trials=1 and verifies that the returned dictionary contains the expected keys:
      - 'clustered_df'
      - 'umap_model'
      - 'hdbscan_model'
      - 'metrics_dict'
    """
    # Construct an absolute path to the test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(test_dir, 'test_embeddings.pkl')
    
    # Read the pickle file as a DataFrame
    df = pd.read_pickle(file_path)
    clustering_engine.n_trials = 1
    result = clustering_engine.optimize(df)
    for key in ("clustered_df", "umap_model", "hdbscan_model", "metrics_dict"):
        assert key in result
