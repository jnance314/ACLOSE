import pytest
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

from atmose.plotting import (
    silhouette_fig,
    bars_fig,
    scatter_fig,
    validate_scatter_params
)
import atmose.plotting as plotting  # for internal function access


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def dummy_clusters_df():
    """
    Create a dummy clusters DataFrame for plotting tests.
    
    The DataFrame includes:
      - 'reduced_vector': A list of coordinates (3D in this case)
      - 'cluster_id': Cluster identifiers (with -1 reserved for noise)
      - 'content': Textual content for each data point
    """
    data = {
        "reduced_vector": [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [1, 0, 0]],
        "cluster_id": [1, 1, 2, -1],
        "content": ["text1", "text2", "text3", "noise"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_missing_cluster_id(dummy_clusters_df):
    """
    Create a DataFrame missing the 'cluster_id' column.
    """
    df = dummy_clusters_df.copy()
    df = df.drop(columns=['cluster_id'])
    return df


@pytest.fixture
def df_missing_reduced_vector(dummy_clusters_df):
    """
    Create a DataFrame missing the 'reduced_vector' column.
    """
    df = dummy_clusters_df.copy()
    df = df.drop(columns=['reduced_vector'])
    return df


@pytest.fixture
def dummy_clusters_single_cluster():
    """
    Create a dummy clusters DataFrame with only one valid (non-noise) cluster.
    Used to test silhouette_fig when insufficient clusters exist.
    """
    data = {
        "reduced_vector": [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]],
        "cluster_id": [1, 1, -1],
        "content": ["text1", "text2", "noise"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_clusters_high_dim():
    """
    Create a dummy clusters DataFrame with 4-dimensional reduced vectors.
    This is used to test the scatter_fig behavior when dims > 3.
    """
    data = {
        "reduced_vector": [[0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], [1, 0, 0, 1]],
        "cluster_id": [1, 1, 2, -1],
        "content": ["text1", "text2", "text3", "noise"]
    }
    return pd.DataFrame(data)


# ------------------------------------------------------------------------------
# Tests for internal functions
# ------------------------------------------------------------------------------

def test_color_map(dummy_clusters_df):
    """
    Test that _color_map returns a mapping for only non-noise clusters.
    """
    color_map = plotting._color_map(dummy_clusters_df)
    # Ensure that noise cluster (-1) is not in the keys
    assert all(cluster != -1 for cluster in color_map.keys())
    unique_clusters = set(dummy_clusters_df["cluster_id"]) - {-1}
    assert set(color_map.keys()) == unique_clusters


def test_color_map_missing_cluster_id(df_missing_cluster_id):
    """
    Test that _color_map raises ValueError when 'cluster_id' is missing.
    """
    with pytest.raises(ValueError, match="DataFrame must contain 'cluster_id' column."):
        plotting._color_map(df_missing_cluster_id)


def test_prepare_topic_counts():
    """
    Test _prepare_topic_counts builds the topic counts DataFrame correctly.
    """
    df = pd.DataFrame({
        "cluster_id": [1, 1, 2, 2, 2, -1],
        "topic": ["A", "A", "B", "B", "B", "Noise"],
        "content": ["a", "b", "c", "d", "e", "noise"]
    })
    topic_counts = plotting._prepare_topic_counts(df)
    # Check that noise (-1) is filtered out
    assert -1 not in topic_counts["cluster_id"].unique()
    # Verify that the count for cluster 1 is 2
    count_cluster1 = topic_counts[topic_counts["cluster_id"] == 1]["count"].iloc[0]
    assert count_cluster1 == 2


def test_prepare_topic_counts_without_topic():
    """
    Test _prepare_topic_counts when the 'topic' column is absent.
    """
    df = pd.DataFrame({
        "cluster_id": [1, 1, 2, 2, 2, -1],
        "content": ["a", "b", "c", "d", "e", "noise"]
    })
    topic_counts = plotting._prepare_topic_counts(df)
    # Check that a default 'topic' column has been added
    assert 'topic' in topic_counts.columns
    expected_topics = topic_counts['cluster_id'].apply(lambda c: f"Cluster {c}")
    # Ignore the difference in Series name by setting check_names=False.
    pd.testing.assert_series_equal(
        topic_counts['topic'].reset_index(drop=True),
        expected_topics.reset_index(drop=True),
        check_names=False
    )


def test_wrap_text_english():
    """
    Test the _wrap_text function on English text.
    """
    text = "This is a long sentence that should be wrapped properly."
    wrapped = plotting._wrap_text(text, width=20)
    # Check that line breaks (<br>) were inserted
    assert "<br>" in wrapped


def test_wrap_text_chinese():
    """
    Test the _wrap_text function on Chinese text.
    """
    text = "这是一个很长的句子应该被正确换行"
    wrapped = plotting._wrap_text(text, width=5)
    # Check that line breaks (<br>) were inserted
    assert "<br>" in wrapped


# ------------------------------------------------------------------------------
# Tests for public interface functions
# ------------------------------------------------------------------------------

def test_silhouette_fig(dummy_clusters_df):
    """
    Test silhouette_fig returns a Plotly Figure with the average silhouette line.
    """
    fig = silhouette_fig(dummy_clusters_df)
    assert isinstance(fig, go.Figure)
    # Check that there is at least one line shape (the average silhouette score indicator)
    # Use shape.type attribute to access type
    assert any(shape.type == 'line' for shape in fig.layout.shapes)


def test_silhouette_fig_missing_reduced_vector(df_missing_reduced_vector):
    """
    Test that silhouette_fig raises ValueError when 'reduced_vector' is missing.
    """
    with pytest.raises(ValueError, match="No 'reduced_vector' column found in DataFrame"):
        silhouette_fig(df_missing_reduced_vector)


def test_silhouette_fig_insufficient_clusters(dummy_clusters_single_cluster):
    """
    Test that silhouette_fig raises ValueError if fewer than 2 valid clusters exist.
    """
    with pytest.raises(ValueError, match="Need at least 2 valid clusters"):
        silhouette_fig(dummy_clusters_single_cluster)


def test_bars_fig(dummy_clusters_df):
    """
    Test that bars_fig returns a valid horizontal bar chart.
    """
    fig = bars_fig(dummy_clusters_df)
    assert isinstance(fig, go.Figure)
    bar_traces = [trace for trace in fig.data if trace.type == "bar"]
    assert len(bar_traces) >= 1


def test_scatter_fig(dummy_clusters_df):
    """
    Test that scatter_fig returns a valid scatter (or scatter3d) plot.
    """
    fig = scatter_fig(dummy_clusters_df, content_col_name="content", wrap_width=50)
    assert isinstance(fig, go.Figure)
    scatter_traces = [trace for trace in fig.data if trace.type in ["scatter", "scatter3d"]]
    assert len(scatter_traces) >= 1


def test_validate_scatter_params_invalid_content(dummy_clusters_df):
    """
    Test validate_scatter_params raises ValueError when content_col_name is missing.
    """
    with pytest.raises(ValueError, match="content_col_name 'missing_content' not found"):
        validate_scatter_params(dummy_clusters_df, "missing_content", 50, None)


def test_validate_scatter_params_invalid_id(dummy_clusters_df):
    """
    Test validate_scatter_params raises ValueError when id_col_name is provided but missing.
    """
    with pytest.raises(ValueError, match="id_col_name 'missing_id' not found"):
        validate_scatter_params(dummy_clusters_df, "content", 50, "missing_id")


def test_validate_scatter_params_wrap_width_too_small(dummy_clusters_df):
    """
    Test that validate_scatter_params defaults wrap_width to at least 20 if a smaller value is given.
    """
    params = validate_scatter_params(dummy_clusters_df, "content", 10, None)
    assert params["wrap_width"] == 20


def test_scatter_fig_with_id(dummy_clusters_df):
    """
    Test scatter_fig when id_col_name is provided.
    The hover text should include the record ID.
    """
    df = dummy_clusters_df.copy()
    df['id'] = [101, 102, 103, 104]
    fig = scatter_fig(df, content_col_name="content", wrap_width=50, id_col_name="id")
    hover_texts = []
    for trace in fig.data:
        t = getattr(trace, 'text', None)
        if t is not None:
            if isinstance(t, (list, np.ndarray)):
                # Convert to list to avoid ambiguity in truth value testing.
                hover_texts.extend(list(t))
            else:
                hover_texts.append(t)
    # Check that at least one hover text contains the record ID.
    assert any("ID:" in text for text in hover_texts)


def test_scatter_fig_high_dim(dummy_clusters_high_dim):
    """
    Test scatter_fig for high-dimensional vectors (>3).
    It should return a figure with an annotation indicating that plotting was skipped.
    """
    fig = scatter_fig(dummy_clusters_high_dim, content_col_name="content", wrap_width=50)
    assert isinstance(fig, go.Figure)
    annotations = fig.layout.annotations
    # At least one annotation should mention that dims > 3 cannot be plotted
    assert any("Cannot plot scatter plot" in annotation.text for annotation in annotations)
