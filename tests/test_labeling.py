import os
import json
import pytest
import pandas as pd
import asyncio
import logging

from atmose.labeling import LabelingEngine, add_labels

# Set dummy API keys so that LabelingEngine does not raise during initialization.
os.environ["OPENAI_API_KEY"] = "dummy_openai_key"
os.environ["HELICONE_API_KEY"] = "dummy_helicone_key"

# A simple dummy response class for simulating async API responses.
class DummyResponse:
    class DummyChoice:
        def __init__(self, message_content):
            self.message = type("DummyMessage", (), {"content": message_content})
    def __init__(self, content):
        self.choices = [self.DummyChoice(content)]

@pytest.fixture
def labeling_engine():
    """
    Fixture to create a LabelingEngine instance configured with dummy parameters.
    
    This instance is used for testing the LabelingEngine's helper functions without invoking real API calls.
    """
    engine = LabelingEngine(
        llm_model='dummy_model',
        language='english',
        ascending=False,
        core_top_n=2,
        peripheral_n=2,
        llm_temp=1.0,
        num_strata=2,
        content_col_name='content',
        data_description="Dummy description"
    )
    return engine

def test_make_model_args(labeling_engine):
    """
    Test the _make_model_args method of LabelingEngine.
    
    The test verifies that:
      - The returned dictionary includes required keys such as 'model', 'temperature', and 'messages'.
      - When Helicone tracing is enabled, the extra_headers dictionary is present and correctly configured.
    
    This ensures that the model arguments are built as expected for API calls.
    """
    system_prompt = "Test prompt"
    core_points = ["text1", "text2"]
    other_centroids = ["centroid1"]
    args = labeling_engine._make_model_args(system_prompt, core_points, other_centroids, func="test_func")
    assert "model" in args
    assert args["model"] == labeling_engine.llm_model
    assert "temperature" in args
    assert "messages" in args
    # If tracing is enabled, check that extra_headers are present.
    if labeling_engine.hcone_trace:
        assert "extra_headers" in args
        headers = args["extra_headers"]
        assert "Helicone-Auth" in headers
        assert headers["Helicone-Property-Function"] == "test_func"

def test_get_centroid_text(labeling_engine):
    """
    Test get_centroid_text to ensure that it correctly selects the text from the row
    with the highest membership_strength among the core points of a cluster.
    
    A dummy DataFrame with two rows in the same cluster is provided; the row with the higher
    membership_strength should be returned.
    """
    df = pd.DataFrame({
        "cluster_id": [1, 1],
        "content": ["text_a", "text_b"],
        "membership_strength": [0.9, 0.8],
        "core_point": [True, True]
    })
    centroid = labeling_engine.get_centroid_text(df, 1)
    assert centroid == "text_a"

def test_get_peripheral_points(labeling_engine):
    """
    Test get_peripheral_points to verify that it returns a sufficient number of peripheral texts.
    
    Given a dummy DataFrame with only non-core points for a cluster, and with the configuration
    set (peripheral_n=2 and num_strata=2), the method should return at least two samples.
    """
    df = pd.DataFrame({
        "cluster_id": [1, 1, 1],
        "content": ["p1", "p2", "p3"],
        "membership_strength": [0.5, 0.4, 0.3],
        "core_point": [False, False, False]
    })
    points = labeling_engine.get_peripheral_points(df, 1)
    assert len(points) >= 2

def test_add_labels_to_cluster_df(labeling_engine):
    """
    Test add_labels_to_cluster_df to check that topic labels are properly assigned to clusters.
    
    This test creates a dummy DataFrame with a 'cluster_id' column and then verifies that:
      - Noise points (cluster_id == -1) are labeled "Noise".
      - Other clusters receive the provided label.
    """
    df = pd.DataFrame({
        "cluster_id": [1, -1, 2],
        "content": ["a", "b", "c"]
    })
    labels = {1: "Label1", 2: "Label2"}
    labeled_df = labeling_engine.add_labels_to_cluster_df(df, labels)
    assert "topic" in labeled_df.columns
    noise_label = labeled_df.loc[labeled_df["cluster_id"] == -1, "topic"].iloc[0]
    assert noise_label == "Noise"
    assert labeled_df.loc[labeled_df["cluster_id"] == 1, "topic"].iloc[0] == "Label1"

@pytest.mark.asyncio
async def test_assign_topic_to_core_points(labeling_engine, monkeypatch):
    """
    Test the asynchronous assign_topic_to_core_points method.
    
    This test monkeypatches the async API call so that it returns a dummy JSON response.
    It then verifies that the method correctly parses the JSON and returns the expected label.
    """
    async def dummy_create(*args, **kwargs):
        response_content = json.dumps({"step_5_final_target_label": "TestLabel"})
        return DummyResponse(response_content)
    labeling_engine.async_oai_client.chat = type("DummyChat", (), {
        "completions": type("DummyCompletions", (), {"create": dummy_create})
    })
    label = await labeling_engine.assign_topic_to_core_points(
        core_points=["core1", "core2"],
        other_centroids=["centroid1"]
    )
    assert label == "TestLabel"

@pytest.mark.asyncio
async def test_generalized_label(labeling_engine, monkeypatch):
    """
    Test the asynchronous generalized_label method for refining topic labels.
    
    The test monkeypatches the async API call to return a dummy JSON response and verifies that
    the method extracts and returns the updated label ("GeneralizedLabel").
    """
    async def dummy_create(*args, **kwargs):
        response_content = json.dumps({"step_5_final_target_label": "GeneralizedLabel"})
        return DummyResponse(response_content)
    labeling_engine.async_oai_client.chat = type("DummyChat", (), {
        "completions": type("DummyCompletions", (), {"create": dummy_create})
    })
    label = await labeling_engine.generalized_label(
        core_points=["core1"],
        core_label="InitialLabel",
        peripheral_points=["peripheral1"],
        other_centroids=["centroid1"]
    )
    assert label == "GeneralizedLabel"
