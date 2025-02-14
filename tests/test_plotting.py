import pytest
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

from atmose.plotting import silhouette_fig, bars_fig, scatter_fig
import atmose.plotting as plotting

@pytest.fixture
def dummy_clusters_df():
    """
    Create a dummy clusters DataFrame for plotting tests.
    
    The DataFrame includes:
      - 'reduced_vector': A list of coordinates (3D in this case)
      - 'cluster_id': Cluster identifiers (with -1 reserved for noise)
      - 'content': Textual content for each data point
    
    This fixture is used to simulate clustering output for plotting functions.
    """
    data = {
        "reduced_vector": [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [1, 0, 0]],
        "cluster_id": [1, 1, 2, -1],
        "content": ["text1", "text2", "text3", "noise"]
    }
    return pd.DataFrame(data)

def test_color_map(dummy_clusters_df):
    """
    Test the _color_map method to ensure that it returns a mapping for only non-noise clusters.
    
    The test verifies that:
      - No noise cluster (-1) is included in the color map.
      - The keys of the returned color map exactly match the set of non-noise clusters.
    """
    color_map = plotting._color_map(dummy_clusters_df)
    assert all(cluster != -1 for cluster in color_map.keys())
    unique_clusters = set(dummy_clusters_df["cluster_id"]) - {-1}
    assert set(color_map.keys()) == unique_clusters

def test_prepare_topic_counts():
    """
    Test _prepare_topic_counts to verify that the topic counts DataFrame is correctly built.
    
    The test:
      - Provides a dummy DataFrame with a 'topic' column.
      - Checks that noise points (cluster_id == -1) are filtered out.
      - Verifies that the count for a given cluster is computed correctly.
    """
    df = pd.DataFrame({
        "cluster_id": [1, 1, 2, 2, 2, -1],
        "topic": ["A", "A", "B", "B", "B", "Noise"],
        "content": ["a", "b", "c", "d", "e", "noise"]
    })
    topic_counts = plotting._prepare_topic_counts(df)
    assert -1 not in topic_counts["cluster_id"].unique()
    count_cluster1 = topic_counts[topic_counts["cluster_id"] == 1]["count"].iloc[0]
    assert count_cluster1 == 2

def test_silhouette_fig(dummy_clusters_df):
    """
    Test the silhouette_fig function to verify that it returns a Plotly Figure.
    
    The test ensures that:
      - The returned object is an instance of go.Figure.
      - At least one shape is present in the layout (e.g., the average silhouette line).
    """
    fig = silhouette_fig(dummy_clusters_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) >= 1  # type: ignore

def test_bars_fig(dummy_clusters_df):
    """
    Test the bars_fig function to verify that it returns a valid horizontal bar chart.
    
    The test checks that:
      - The returned object is a Plotly Figure.
      - There is at least one bar trace present in the figure.
    """
    fig = bars_fig(dummy_clusters_df)
    assert isinstance(fig, go.Figure)
    bar_traces = [trace for trace in fig.data if trace.type == "bar"]  # type: ignore
    assert len(bar_traces) >= 1

def test_scatter_fig(dummy_clusters_df):
    """
    Test the scatter_fig function to verify that it returns a valid scatter (or scatter3d) plot.
    
    The test ensures that:
      - The returned object is a Plotly Figure.
      - At least one trace of type "scatter" or "scatter3d" is present in the figure.
    """
    fig = scatter_fig(dummy_clusters_df, content_col_name="content", wrap_width=50)
    assert isinstance(fig, go.Figure)
    scatter_traces = [trace for trace in fig.data if trace.type in ["scatter", "scatter3d"]]  # type: ignore
    assert len(scatter_traces) >= 1
