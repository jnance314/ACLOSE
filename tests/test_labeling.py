import os
import json
import pytest
import pandas as pd
import asyncio
import logging

from aclose.labeling import (
    LabelingEngine,
    add_labels,
    validate_labeling_params,
    LLMSettings,
    SamplingSettings,
)

# Set dummy API keys so that LabelingEngine does not raise during initialization.
os.environ["OPENAI_API_KEY"] = "dummy_openai_key"
os.environ["HELICONE_API_KEY"] = "dummy_helicone_key"

# -------------------------------
# Dummy Response & Monkeypatch Helpers
# -------------------------------


class DummyResponse:
    class DummyChoice:
        def __init__(self, message_content, finish_reason="stop"):
            self.message = type("DummyMessage", (), {"content": message_content})
            self.finish_reason = finish_reason

    def __init__(self, content, finish_reason="stop"):
        self.choices = [self.DummyChoice(content, finish_reason)]


# Dummy asynchronous methods for generating/updating topics.
def dummy_generate_initial_topics_async(self, cluster_df):
    async def _dummy():
        # For each unique cluster_id (except noise), assign a label.
        unique_clusters = cluster_df[cluster_df["cluster_id"] != -1][
            "cluster_id"
        ].unique()
        return {int(cluster): f"Label{cluster}" for cluster in unique_clusters}

    return _dummy()


def dummy_update_topics_async(self, cluster_df, initial_topics):
    async def _dummy():
        # For simplicity, return the initial topics without change.
        return initial_topics

    return _dummy()


# -------------------------------
# Pytest Fixtures
# -------------------------------


@pytest.fixture
def labeling_engine():
    """
    Fixture to create a LabelingEngine instance configured with dummy parameters.
    This instance is used for testing helper functions without invoking real API calls.
    """
    llm_settings = LLMSettings(
        model="dummy_model",
        language="english",
        temperature=1.0,
        data_description="Dummy description",
    )
    sampling_settings = SamplingSettings(
        ascending=False,
        core_top_n=2,
        peripheral_n=2,
        num_strata=2,
        content_col_name="content",
    )
    engine = LabelingEngine(
        llm_settings=llm_settings, sampling_settings=sampling_settings
    )
    return engine


# -------------------------------
# Tests for Helper Methods
# -------------------------------


def test_make_model_args(labeling_engine):
    """
    Test the _make_model_args method of LabelingEngine.
    Verifies that required keys are present and if Helicone tracing is enabled,
    extra_headers are correctly configured.
    """
    system_prompt = "Test prompt"
    core_points = ["text1", "text2"]
    other_centroids = ["centroid1"]
    args = labeling_engine._make_model_args(
        system_prompt,
        core_points,
        other_centroids,
        core_label=None,
        peripheral_points=None,
        func="topic-label-pass-2",
    )
    assert "model" in args
    assert args["model"] == labeling_engine.llm_model
    assert "temperature" in args
    assert "messages" in args
    if labeling_engine.hcone_trace:
        assert "extra_headers" in args
        headers = args["extra_headers"]
        assert "Helicone-Auth" in headers
        assert headers["Helicone-Property-Function"] == "topic-label-pass-2"


def test_get_centroid_text(labeling_engine):
    """
    Test that get_centroid_text returns the text with the highest membership_strength among core points.
    """
    df = pd.DataFrame(
        {
            "cluster_id": [1, 1],
            "content": ["text_a", "text_b"],
            "membership_strength": [0.9, 0.8],
            "core_point": [True, True],
        }
    )
    centroid = labeling_engine.get_centroid_text(df, 1)
    assert centroid == "text_a"


def test_get_centroid_text_no_core(labeling_engine):
    """
    Test that get_centroid_text falls back to the first available text when no core points exist.
    """
    df = pd.DataFrame(
        {
            "cluster_id": [1, 1],
            "content": ["text_a", "text_b"],
            "membership_strength": [0.8, 0.9],
            "core_point": [False, False],
        }
    )
    centroid = labeling_engine.get_centroid_text(df, 1)
    # Should fall back to the first row.
    assert centroid == "text_a"


def test_get_peripheral_points(labeling_engine):
    """
    Test get_peripheral_points returns a sufficient number of peripheral texts.
    """
    df = pd.DataFrame(
        {
            "cluster_id": [1, 1, 1],
            "content": ["p1", "p2", "p3"],
            "membership_strength": [0.5, 0.4, 0.3],
            "core_point": [False, False, False],
        }
    )
    points = labeling_engine.get_peripheral_points(df, 1)
    # The number of returned points should be at least the peripheral_n setting.
    assert len(points) >= labeling_engine.peripheral_n


def test_add_labels_to_cluster_df(labeling_engine):
    """
    Test that add_labels_to_cluster_df assigns topic labels properly.
    """
    df = pd.DataFrame({"cluster_id": [1, -1, 2], "content": ["a", "b", "c"]})
    labels = {1: "Label1", 2: "Label2"}
    labeled_df = labeling_engine.add_labels_to_cluster_df(df, labels)
    assert "topic" in labeled_df.columns
    noise_label = labeled_df.loc[labeled_df["cluster_id"] == -1, "topic"].iloc[0]
    assert noise_label == "Noise"
    assert labeled_df.loc[labeled_df["cluster_id"] == 1, "topic"].iloc[0] == "Label1"


# -------------------------------
# Tests for Asynchronous Methods
# -------------------------------


@pytest.mark.asyncio
async def test_assign_topic_to_core_points(labeling_engine, monkeypatch):
    """
    Test the asynchronous assign_topic_to_core_points method by monkeypatching the async API call.
    It simulates a dummy JSON response and verifies the label extraction.
    """

    async def dummy_create(*args, **kwargs):
        response_content = json.dumps({"final_target_label": "TestLabel"})
        return DummyResponse(response_content)

    labeling_engine.async_oai_client.chat = type(
        "DummyChat",
        (),
        {"completions": type("DummyCompletions", (), {"create": dummy_create})},
    )
    label = await labeling_engine.assign_topic_to_core_points(
        core_points=["core1", "core2"], other_centroids=["centroid1"]
    )
    assert label == "TestLabel"


@pytest.mark.asyncio
async def test_generalized_label(labeling_engine, monkeypatch):
    """
    Test the asynchronous generalized_label method by monkeypatching the async API call.
    It simulates a dummy JSON response and verifies that the updated label is returned.
    """

    async def dummy_create(*args, **kwargs):
        response_content = json.dumps({"final_target_label": "GeneralizedLabel"})
        return DummyResponse(response_content)

    labeling_engine.async_oai_client.chat = type(
        "DummyChat",
        (),
        {"completions": type("DummyCompletions", (), {"create": dummy_create})},
    )
    label = await labeling_engine.generalized_label(
        core_points=["core1"],
        core_label="InitialLabel",
        peripheral_points=["peripheral1"],
        other_centroids=["centroid1"],
    )
    assert label == "GeneralizedLabel"


# -------------------------------
# Tests for Parameter Validation
# -------------------------------


def test_validate_labeling_params_valid():
    """
    Test validate_labeling_params with valid parameters.
    """
    df = pd.DataFrame(
        {
            "content": ["sample"],
            "cluster_id": [1],
            "membership_strength": [0.5],
            "core_point": [True],
        }
    )
    result = validate_labeling_params(
        cluster_df=df,
        llm_model="model1",
        language="english",
        temperature=0.5,
        data_description="Test description",
        ascending=True,
        core_top_n=5,
        peripheral_n=5,
        num_strata=2,
        content_col_name="content",
    )
    assert result is True


def test_validate_labeling_params_invalid():
    """
    Test validate_labeling_params with invalid parameters to ensure it raises ValueError.
    """
    df = "not a dataframe"
    with pytest.raises(ValueError):
        validate_labeling_params(
            cluster_df=df,
            llm_model="model1",
            language="english",
            temperature=0.5,
            data_description="Test description",
            ascending=True,
            core_top_n=5,
            peripheral_n=5,
            num_strata=2,
            content_col_name="content",
        )


# -------------------------------
# Test for High-level API Function
# -------------------------------


def test_add_labels_function(monkeypatch):
    """
    Test the high-level add_labels function.
    Monkeypatch the asynchronous methods in LabelingEngine to return dummy labels.
    """
    monkeypatch.setattr(
        LabelingEngine,
        "generate_initial_topics_async",
        dummy_generate_initial_topics_async,
    )
    monkeypatch.setattr(
        LabelingEngine, "update_topics_async", dummy_update_topics_async
    )

    # Create a dummy DataFrame for clusters.
    df = pd.DataFrame(
        {
            "cluster_id": [1, 1, 2, -1],
            "content": ["text1", "text2", "text3", "noise_text"],
            "membership_strength": [0.9, 0.8, 0.7, 0.5],
            "core_point": [True, False, True, False],
        }
    )

    results = add_labels(
        cluster_df=df,
        llm_model="dummy_model",
        language="english",
        temperature=1.0,
        data_description="Dummy description",
        ascending=False,
        core_top_n=1,
        peripheral_n=1,
        num_strata=1,
        content_col_name="content",
    )

    labeled_df = results["dataframe"]
    labels_dict = results["labels_dict"]

    # Verify that the dataframe now contains the 'topic' column.
    assert "topic" in labeled_df.columns
    # Check that noise points are labeled as 'Noise'
    noise_label = labeled_df.loc[labeled_df["cluster_id"] == -1, "topic"].iloc[0]
    assert noise_label == "Noise"
    # Verify that labels_dict contains correct mappings for clusters (excluding noise).
    for cluster in labeled_df["cluster_id"].unique():
        if cluster == -1:
            continue
        assert cluster in labels_dict
        # The dummy function assigns labels as "Label{cluster}"
        assert labels_dict[cluster] == f"Label{cluster}"
