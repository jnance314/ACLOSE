import pandas as pd
import json
import asyncio
import os
import logging
import functools
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI
from typing import Dict

# -------------------------------
# Configuration Dataclasses
# -------------------------------


@dataclass
class LLMSettings:
    """
    Configuration settings for the language model.

    Attributes:
        model (str): Identifier for the LLM model to use (e.g., 'o1').
        language (str): The language for the output labels (e.g., 'english').
        temperature (float): Temperature parameter for the language model.
        data_description (str): Additional description of the data to be appended to the prompt.
    """

    model: str = "o1"
    language: str = "english"
    temperature: float = 1.0
    data_description: str = (
        "No data description provided. Just do your best to infer/assume the context of the data while performing your tasks."
    )


@dataclass
class SamplingSettings:
    """
    Configuration settings for cluster sampling and processing.

    Attributes:
        ascending (bool): Order for processing clusters; if True, clusters are processed in ascending order.
        core_top_n (int): Number of top core points to consider for initial labeling.
        peripheral_n (int): Number of peripheral points to consider for refining the label.
        num_strata (int): Number of strata for stratified sampling of peripheral points.
        content_col_name (str): Name of the column containing the text content in the DataFrame.
    """

    ascending: bool = False
    core_top_n: int = 10
    peripheral_n: int = 12
    num_strata: int = 3
    content_col_name: str = "content"


# -------------------------------
# Asynchronous Retry Decorator
# -------------------------------


def async_retry(max_attempts=3, exceptions=(Exception,)):
    """
    Decorator for retrying an asynchronous function call upon specified exceptions.

    Args:
        max_attempts (int): Maximum number of attempts before failing.
        exceptions (tuple): Tuple of exception classes that trigger a retry.

    Returns:
        The result of the asynchronous function if successful.

    Raises:
        Exception: If the function fails after the maximum number of attempts.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt_count = 0
            while attempt_count < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt_count += 1
                    # Use the instance's logger if available
                    logger = getattr(args[0], "logger", None)
                    if logger:
                        logger.warning(
                            "Attempt %d/%d in %s failed: %s",
                            attempt_count,
                            max_attempts,
                            func.__name__,
                            str(e),
                        )
                    if attempt_count >= max_attempts:
                        raise Exception(
                            f"Failed to process response after {max_attempts} attempts"
                        ) from e

        return wrapper

    return decorator


# -------------------------------
# LabelingEngine Class
# -------------------------------


class LabelingEngine:
    def __init__(self, llm_settings: LLMSettings, sampling_settings: SamplingSettings):
        """
        Initialize the LabelingEngine instance.

        This class provides methods for generating semantic topic labels for clusters using a language model.
        It supports both initial topic generation from core texts and topic refinement by sampling peripheral texts.

        Args:
            llm_settings (LLMSettings): Configuration settings for the language model.
            sampling_settings (SamplingSettings): Configuration settings for cluster sampling and processing.

        Raises:
            ValueError: If the required environment variables (OPENAI_API_KEY or HELICONE_API_KEY) are not set.
        """
        # -------------------------------
        # Save configuration parameters as instance attributes.
        # -------------------------------
        self.llm_model: str = llm_settings.model
        self.language: str = llm_settings.language
        self.llm_temp: float = llm_settings.temperature
        self.data_description: str = llm_settings.data_description

        self.ascending: bool = sampling_settings.ascending
        self.core_top_n: int = sampling_settings.core_top_n
        self.peripheral_n: int = sampling_settings.peripheral_n
        self.num_strata: int = sampling_settings.num_strata
        self.content_col_name: str = sampling_settings.content_col_name

        # -------------------------------
        # Initialize Logger for this instance.
        # -------------------------------
        self.logger = logging.getLogger(self.__class__.__name__)

        # NOTE: The debug message below references self.hcone_trace,
        # which is set later. This message will only be fully correct after key retrieval.
        self.logger.debug(
            "LabelingEngine initialized with llm_model=%s, language=%s, ascending=%s, core_top_n=%d, peripheral_n=%d, hcone_trace=%s",
            self.llm_model,
            self.language,
            self.ascending,
            self.core_top_n,
            self.peripheral_n,
            "Not set yet",  # Placeholder since self.hcone_trace is defined later
        )

        # -------------------------------
        # Retrieve required API keys from environment variables.
        # -------------------------------
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")
        self.hcone_trace = True  # Default to True; may be disabled if key not available

        if not self.OPENAI_API_KEY:
            self.logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        if not self.HELICONE_API_KEY:
            self.logger.error("HELICONE_API_KEY environment variable not set.")
            self.hcone_trace = False
            raise ValueError(
                "HELICONE_API_KEY environment variable not set. Disabling tracing."
            )
        self.logger.debug("API keys retrieved successfully.")

        # -------------------------------
        # Initialize OpenAI clients:
        # - Synchronous client for non-async calls.
        # - Asynchronous client for async operations.
        # -------------------------------
        self.oai_client = OpenAI(
            api_key=self.OPENAI_API_KEY, base_url="http://oai.hconeai.com/v1"
        )
        self.async_oai_client = AsyncOpenAI(
            api_key=self.OPENAI_API_KEY, base_url="http://oai.hconeai.com/v1"
        )

    def _make_model_args(
        self,
        system_prompt,
        core_points,
        other_centroids,
        core_label,
        peripheral_points,
        func,
    ):
        """
        Construct the model arguments for an API call to generate or refine a topic label.

        This method builds a payload dictionary for the language model API based on the provided system prompt,
        core texts, and centroid texts from other clusters. When `func` is 'topic-label-pass-1', only `core_points`
        and `other_centroids` are used. When `func` is 'topic-label-pass-2', the `core_label` and `peripheral_points` are
        also incorporated to refine the initial label. If Helicone tracing is enabled, appropriate tracing headers are added.

        Args:
            system_prompt (str): The system prompt detailing the task and instructions.
            core_points (list[str]): List of core texts from the target cluster.
            other_centroids (list[str]): List of centroid texts from other clusters.
            core_label (str or None): The initial topic label (used only for label refinement).
            peripheral_points (list or None): List of peripheral texts (used only for label refinement).
            func (str): Identifier for the API call type ('topic-label-pass-1' or 'topic-label-pass-2').

        Returns:
            dict: The dictionary of model arguments tailored for the specified API call.
        """
        self.logger.debug("Building model arguments for API call.")

        # -------------------------------
        # Construct the basic payload for the API call.
        # -------------------------------
        model_args = {
            "topic-label-pass-1": {
                "model": self.llm_model,
                "temperature": self.llm_temp,
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
                        {system_prompt}
                        
                        Core texts from target cluster:
                        {core_points}
        
                        Centroid texts from other clusters:
                        {other_centroids}
                    """,
                    }
                ],
            },
            "topic-label-pass-2": {
                "model": self.llm_model,
                "temperature": self.llm_temp,
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
                        {system_prompt}
                        
                        Core texts from target cluster:
                        {core_points} 
                        
                        Target's label, considering only core points:
                        {core_label}
                        
                        Peripheral texts from target cluster:
                        {peripheral_points} 
                        
                        Centroid texts from other clusters:
                        {other_centroids}
                    """,
                    }
                ],
            },
        }

        # -------------------------------
        # Add tracing headers if Helicone tracing is enabled.
        # -------------------------------
        if self.hcone_trace:
            model_args[func]["extra_headers"] = {
                "Helicone-Auth": f"Bearer {self.HELICONE_API_KEY}",
                "Helicone-Retry-Enabled": "true",
                "Helicone-Property-Function": func,
            }
        self.logger.debug(
            "Model arguments constructed with model: %s and system_prompt (first 100 chars): %s",
            self.llm_model,
            system_prompt.strip()[:100],
        )
        return model_args[func]

    @async_retry(max_attempts=3, exceptions=(json.JSONDecodeError, KeyError, Exception))
    async def assign_topic_to_core_points(
        self, core_points: list, other_centroids: list
    ) -> str | None:
        """
        Generate an initial topic label for a cluster using its core points.

        This asynchronous method calls the language model to produce an initial topic label based on a list of
        core texts from the target cluster and centroid texts from other clusters for contrast. It raises a
        ValueError if either `core_points` or `other_centroids` is empty.

        Args:
            core_points (list[str]): List of core texts from the target cluster.
            other_centroids (list[str]): List of centroid texts from other clusters.

        Returns:
            str: The final topic label as determined by the language model.

        Raises:
            ValueError: If `core_points` or `other_centroids` is empty.
            Exception: If the API call fails after the maximum number of retry attempts.
        """
        # raise exception if core_points or other_centroids is empty
        if not core_points:
            raise ValueError(
                "core_points cannot be empty. It's possible you are passing a df that was clustered with too low a hdbscan_outlier_threshold which resulted in no core_points being found."
            )
        if not other_centroids:
            raise ValueError("other_centroids cannot be empty.")

        self.logger.info("Starting topic labeling...")

        # -------------------------------
        # Reminder text to enforce JSON response format.
        # -------------------------------
        json_reminder = """
        Your response should always begin with
        ```json
        {{
            "final_target_label":
        """

        # -------------------------------
        # Build the detailed system prompt with task instructions.
        # -------------------------------
        system_prompt = f"""
        ## Task:
        - A large corpus of text excerpts has been embedded using transformer-based language models. The vectors have gone through dimensionality reduction and clustering, and we are now trying to assign a topic to each cluster.
        - A collection of core points with high membership scores from a single cluster is provided, along with the texts corresponding to the centroids of the other clusters. The end goal is to assign a topic label to the cluster in question that best represents the cluster, while at the same time differentiating it from the other clusters' centroid texts.
        - However, we will not simply assign a label in one step. Instead, we will reason in steps, finally arriving at a label after thinking out loud a couple times.
        - You will respond as valid JSON in a schema described below. DO NOT BE CONVERSATIONAL IN YOUR RESPONSE. Instead, respond only as a single JSON object as described in the schema.

        ## Description of the data:
        {self.data_description}
    
        ## What makes a good topic label?
        - A good topic label should be specific enough to differentiate the cluster from others, but general enough to encompass the core points in the cluster.
        - It should be a noun or noun phrase that describes the main theme or topic of the messages in the cluster.
        - It should not be too specific or too general – we want to address the bias-variance tradeoff. The label should fit the examples well, while also being general enough to apply to new examples.
        - To help with the specificity-generalization tradeoff, texts corresponding to the centroids of other clusters are provided. The label should be specific enough to distinguish the target cluster from these.
    
        ## Five-step process:
        - step 1: Think out loud about the themes or topics in the representative texts of the target cluster.
        - step 2: Consider what makes the target cluster distinct from the other clusters.
        - step 3: Identify good candidate themes.
        - step 4: Propose a few candidate labels.
        - step 5: Finally, choose the best label (10 words or less).
    
        ## JSON Schema:
        {{
            "final_target_label": <label in 10 words or less>
        }}
    
        DO NOT BE CONVERSATIONAL IN YOUR RESPONSE. Instead, respond only as a single JSON object as described above.
    
        {json_reminder if self.llm_model in ['o1', 'o1-preview', 'o1-mini'] else ""}
        """
        # -------------------------------
        # Make the asynchronous API call to the language model.
        # -------------------------------
        response = await self.async_oai_client.chat.completions.create(
            **self._make_model_args(
                system_prompt=system_prompt,
                core_points=core_points,
                other_centroids=other_centroids,
                func="topic-label-pass-1",
                core_label=None,
                peripheral_points=None,
            )
        )
        self.logger.debug("Received response from async_oai_client.")

        # -------------------------------
        # log a warning if response.choices[0].finish_reason != 'stop', and echo it.
        # -------------------------------
        finish_reason = response.choices[0].finish_reason
        if finish_reason != "stop":
            self.logger.warning(
                "WARNING: Response not generated. OpenAI finish_reason was %s",
                finish_reason,
            )

        # -------------------------------
        # Clean up the response: remove newlines and stray text.
        # -------------------------------
        content = (
            str(response.choices[0].message.content)
            .replace("```", "")
            .replace("json", "")
            .strip()
        )
        self.logger.debug("Cleaned response content: %.100s", content)

        # -------------------------------
        # Parse the JSON content and extract the final label.
        # -------------------------------
        parsed = json.loads(content)
        label = parsed["final_target_label"]
        self.logger.debug("Parsed label: %s", label)
        return label

    @async_retry(max_attempts=3, exceptions=(json.JSONDecodeError, KeyError, Exception))
    async def generalized_label(
        self,
        core_points: list,
        core_label: str,
        peripheral_points: list,
        other_centroids: list,
    ) -> str | None:
        """
        Refine an existing topic label by incorporating peripheral texts.

        This asynchronous method refines the initial topic label by taking into account both the core texts and a
        stratified sample of peripheral texts from the target cluster, along with centroid texts from other clusters
        for additional context. It raises a ValueError if any of the required inputs (`core_points`, `core_label`,
        `peripheral_points`, or `other_centroids`) are empty.

        Args:
            core_points (list[str]): List of core texts from the target cluster.
            core_label (str): The initial topic label generated from core texts.
            peripheral_points (list[str]): List of peripheral texts from the target cluster.
            other_centroids (list[str]): List of centroid texts from other clusters.

        Returns:
            str: The refined topic label as determined by the language model.

        Raises:
            ValueError: If any of the required arguments is empty.
            Exception: If the API call fails after the maximum number of retry attempts.
        """
        self.logger.info("Refining initial topic...")
        # raise exception if any of necessary args are empty
        if not core_points:
            raise ValueError(
                "core_points cannot be empty. It's possible you are passing a df that was clustered with too low a hdbscan_outlier_threshold which resulted in no core_points being found."
            )
        if not core_label:
            raise ValueError("core_label cannot be empty.")
        if not peripheral_points:
            raise ValueError("peripheral_points cannot be empty.")
        if not other_centroids:
            raise ValueError("other_centroids cannot be empty.")

        # -------------------------------
        # Reminder text to enforce JSON response format.
        # -------------------------------
        json_reminder = """
        Your response should always begin with
        ```json
        {{
            "final_target_label":
        """

        # -------------------------------
        # Build the system prompt for updating the label using both core and peripheral texts.
        # -------------------------------
        system_prompt = f"""
        ## Task:
        - A large corpus of text excerpts has been embedded using transformer-based language models. The vectors have gone through dimensionality reduction and clustering, and we are now trying to assign a topic to each cluster.
        - A collection of core points with high membership scores from a single cluster is provided, along with a proposed label for the cluster (based only on some core points). In this task, you are also given a sample of peripheral texts from the target cluster which you will use to update the original label so that it generalizes well.
        - The end goal is to assign a topic label that best represents the entire target cluster, while distinguishing it from the centroids of other clusters.
        - You will respond as valid JSON in a schema described below. DO NOT BE CONVERSATIONAL IN YOUR RESPONSE. Instead, respond only as a single JSON object as described.
    
        {"- REMINDER: You must write your final_target_label in " + self.language if self.language != 'english' else ""}

        ## Description of the data:
        {self.data_description}
    
        ## What makes a good topic label?
        - It should be specific enough to differentiate the cluster yet general enough to cover the cluster’s variation.
        - It should be a noun or noun phrase that describes the main theme.
        - Consider the texts corresponding to the centroids of other clusters when proposing the label.
    
        ## Five-step process:
        - step 1: Discuss the themes in the core texts together with the original label.
        - step 2: Consider if and how the label should be updated given the peripheral texts.
        - step 3: Compare the target cluster to other clusters.
        - step 4: Propose several candidate labels.
        - step 5: Finally, choose the best label (10 words or less).
    
        ## JSON Schema:
        {{
            "final_target_label": <label in 10 words or less>
        }}
        {"- REMINDER: You must write your final_target_label in " + self.language if self.language != 'english' else ""}
        {json_reminder if self.llm_model in ['o1', 'o1-preview', 'o1-mini'] else ""}
        """
        # -------------------------------
        # Make the asynchronous API call to update the label.
        # -------------------------------
        response = await self.async_oai_client.chat.completions.create(
            **self._make_model_args(
                system_prompt=system_prompt,
                core_points=core_points,
                other_centroids=other_centroids,
                func="topic-label-pass-2",
                core_label=core_label,
                peripheral_points=peripheral_points,
            )
        )
        self.logger.debug("Received response for label generalization.")

        # -------------------------------
        # log a warning if response.choices[0].finish_reason != 'stop', and echo it.
        # -------------------------------
        finish_reason = response.choices[0].finish_reason
        if finish_reason != "stop":
            self.logger.warning(
                "WARNING: Response not generated. OpenAI finish_reason was %s",
                finish_reason,
            )

        # -------------------------------
        # Clean and parse the response content.
        # -------------------------------
        content = (
            str(response.choices[0].message.content)
            .replace("```", "")
            .replace("json", "")
            .strip()
        )
        self.logger.debug("Cleaned response content (first 100 chars): %.100s", content)
        parsed = json.loads(content)
        label = parsed["final_target_label"]
        self.logger.debug("Parsed updated label: %s", label)
        return label

    async def generate_initial_topics_async(self, cluster_df: pd.DataFrame) -> dict:
        """
        Generate initial topic labels for each cluster asynchronously.

        This method processes clusters (excluding noise clusters with cluster_id -1) to generate an initial topic label
        for each cluster. For each cluster, it selects the top N core points (sorted by membership strength) and collects
        centroid texts from other clusters for contrast. It then calls the language model asynchronously to generate a label.

        Args:
            cluster_df (pd.DataFrame): DataFrame containing cluster data with columns including 'cluster_id', 'core_point', 'membership_strength', and a text column (specified by content_col_name).

        Returns:
            dict: A dictionary mapping cluster IDs (int) to their initial topic labels (str).
        """
        try:
            self.logger.debug("Generating initial topics asynchronously.")

            # -------------------------------
            # Determine clusters to process (exclude noise) and order them based on configuration.
            # -------------------------------
            cluster_sizes = cluster_df[cluster_df["cluster_id"] != -1][
                "cluster_id"
            ].value_counts()
            self.logger.debug("Cluster sizes (non-noise): %s", cluster_sizes.to_dict())
            clusters = (
                cluster_sizes.index.tolist()
                if self.ascending
                else cluster_sizes.index.tolist()[::-1]
            )

            tasks = []
            # -------------------------------
            # Iterate over each cluster to schedule an asynchronous topic assignment.
            # -------------------------------
            for cluster in clusters:
                if cluster == -1:
                    continue
                self.logger.debug(
                    "Scheduling initial topic assignment for cluster %s.", cluster
                )

                # -------------------------------
                # Select the top core points for the current cluster.
                # -------------------------------
                core_points_df = cluster_df[
                    (cluster_df["cluster_id"] == cluster) & (cluster_df["core_point"])
                ]
                core_points_df = core_points_df.sort_values(
                    by="membership_strength", ascending=False
                )
                core_points_texts = (
                    core_points_df[self.content_col_name].head(self.core_top_n).tolist()
                )

                # -------------------------------
                # Prepare centroid texts from all other clusters for contrast.
                # -------------------------------
                other_clusters = [c for c in clusters if c != cluster and c != -1]
                other_centroids_texts = [
                    self.get_centroid_text(cluster_df, c) for c in other_clusters
                ]

                # -------------------------------
                # Schedule the asynchronous API call for initial topic assignment.
                # -------------------------------
                task = asyncio.create_task(
                    self.assign_topic_to_core_points(
                        core_points=core_points_texts,
                        other_centroids=other_centroids_texts,
                    )
                )
                tasks.append((cluster, task))

            # -------------------------------
            # Await completion of all scheduled tasks.
            # -------------------------------
            self.logger.debug(
                "Waiting for all initial topic assignment tasks to complete."
            )
            results = await asyncio.gather(
                *(t for _, t in tasks), return_exceptions=True
            )

            # -------------------------------
            # Check for and raise any exceptions encountered during processing.
            # -------------------------------
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(
                        "Error during generating initial topics: %s", str(result)
                    )
                    raise result

            # -------------------------------
            # Build a mapping of cluster IDs to their corresponding initial topic labels.
            # -------------------------------
            initial_topics = {
                cluster: label for (cluster, _), label in zip(tasks, results)
            }
            self.logger.debug("Initial topics generated: %s", initial_topics)
            return initial_topics

        except Exception as e:
            self.logger.error(
                "An error occurred in generate_initial_topics_async: %s", str(e)
            )
            raise

    def get_peripheral_points(self, cluster_df: pd.DataFrame, cluster: int) -> list:
        """
        Retrieve stratified peripheral points for a given cluster.

        This method performs stratified sampling on the peripheral (non-core) points of a cluster, using the membership strength
        to assign points to strata. It then samples a fixed number of texts from each stratum to provide a representative
        set of texts for refining the topic label. If the total sampled texts are fewer than requested, additional texts are sampled.

        Args:
            cluster_df (pd.DataFrame): DataFrame containing cluster data.
            cluster (int): The cluster ID for which peripheral points are required.

        Returns:
            list[str]: A list of peripheral point texts.
        """
        self.logger.debug("Retrieving peripheral points for cluster %d.", cluster)

        # -------------------------------
        # Filter the DataFrame for peripheral (non-core) points for the target cluster.
        # -------------------------------
        peripheral_points_df = cluster_df[
            (cluster_df["cluster_id"] == cluster) & (~cluster_df["core_point"])
        ]
        if peripheral_points_df.empty:
            self.logger.debug("No peripheral points found for cluster %d.", cluster)
            return []

        # -------------------------------
        # Work on a copy to avoid modifying the original DataFrame.
        # -------------------------------
        peripheral_points_df = peripheral_points_df.copy()

        # -------------------------------
        # Stratify peripheral points based on their membership strength.
        # -------------------------------
        peripheral_points_df["stratum"] = pd.qcut(
            -peripheral_points_df["membership_strength"],
            q=min(self.num_strata, len(peripheral_points_df)),
            labels=False,
            duplicates="drop",
        )

        peripheral_points_texts = []
        # -------------------------------
        # For each stratum, sample a fixed number of texts.
        # -------------------------------
        for stratum in peripheral_points_df["stratum"].unique():
            stratum_df = peripheral_points_df[
                peripheral_points_df["stratum"] == stratum
            ]
            n_samples_per_stratum = max(1, self.peripheral_n // self.num_strata)
            sampled_texts = (
                stratum_df[self.content_col_name]
                .sample(
                    n=min(n_samples_per_stratum, len(stratum_df)),
                    random_state=42,
                    replace=False,
                )
                .tolist()
            )
            self.logger.debug(
                "Sampled %d peripheral points from stratum %s for cluster %d.",
                len(sampled_texts),
                stratum,
                cluster,
            )
            peripheral_points_texts.extend(sampled_texts)

        # -------------------------------
        # If total samples are insufficient, sample additional texts.
        # -------------------------------
        if len(peripheral_points_texts) < self.peripheral_n:
            additional_needed = self.peripheral_n - len(peripheral_points_texts)
            remaining_points = peripheral_points_df[
                ~peripheral_points_df[self.content_col_name].isin(
                    peripheral_points_texts
                )
            ]
            if not remaining_points.empty:
                additional_texts = (
                    remaining_points[self.content_col_name]
                    .sample(
                        n=min(additional_needed, len(remaining_points)),
                        random_state=42,
                        replace=False,
                    )
                    .tolist()
                )
                peripheral_points_texts.extend(additional_texts)
                self.logger.debug(
                    "Added %d additional peripheral points for cluster %d.",
                    len(additional_texts),
                    cluster,
                )

        self.logger.debug(
            "Total peripheral points collected for cluster %d: %d",
            cluster,
            len(peripheral_points_texts),
        )
        return peripheral_points_texts

    def get_centroid_text(self, cluster_df: pd.DataFrame, cluster: int) -> str:
        """
        Retrieve the centroid text for a given cluster.

        This method selects the text from the core points of a cluster with the highest membership strength, or falls back
        to any available text if no core points are present. The text is taken from the column specified by 'content_col_name'.

        Args:
            cluster_df (pd.DataFrame): DataFrame containing cluster data.
            cluster (int): The cluster ID for which the centroid text is required.

        Returns:
            str: The content of the centroid text.
        """
        self.logger.debug("Retrieving centroid text for cluster %d.", cluster)

        # -------------------------------
        # Filter the DataFrame for the target cluster.
        # -------------------------------
        cluster_data = cluster_df[cluster_df["cluster_id"] == cluster]

        # -------------------------------
        # Prefer core points sorted by membership strength.
        # -------------------------------
        core_points_df = cluster_data[cluster_data["core_point"]]
        core_points_df = core_points_df.sort_values(
            by="membership_strength", ascending=False
        )
        if not core_points_df.empty:
            centroid_point = core_points_df.iloc[0]
        else:
            centroid_point = cluster_data.iloc[0]
        self.logger.debug(
            "Centroid text for cluster %d selected (first 100 chars): %.100s",
            cluster,
            centroid_point[self.content_col_name],
        )
        return centroid_point[self.content_col_name]

    async def update_topics_async(
        self, cluster_df: pd.DataFrame, initial_topics: dict
    ) -> dict:
        """
        Update topic labels for each cluster using peripheral texts asynchronously.

        This method refines the initial topic labels by incorporating stratified samples of peripheral texts along with
        core texts and centroid texts from other clusters. Each cluster is processed asynchronously to produce an updated topic label.
        If an initial topic label is not found for a cluster, a default label "Unknown" is used.

        Args:
            cluster_df (pd.DataFrame): DataFrame containing cluster data.
            initial_topics (dict): Dictionary mapping cluster IDs to their initial topic labels.

        Returns:
            dict: A dictionary mapping cluster IDs (int) to their updated topic labels (str).
        """
        try:
            self.logger.debug("Updating topics asynchronously for clusters.")

            # -------------------------------
            # Determine clusters to process (exclude noise) and order them as configured.
            # -------------------------------
            cluster_sizes = cluster_df[cluster_df["cluster_id"] != -1][
                "cluster_id"
            ].value_counts()
            clusters = (
                cluster_sizes.index.tolist()
                if self.ascending
                else cluster_sizes.index.tolist()[::-1]
            )

            tasks = []
            # -------------------------------
            # For each cluster, schedule an asynchronous label update.
            # -------------------------------
            for cluster in clusters:
                if cluster == -1:
                    continue
                core_label = initial_topics.get(cluster, "Unknown")
                self.logger.debug(
                    "Scheduling topic update for cluster %d with initial label: %s",
                    cluster,
                    core_label,
                )

                # -------------------------------
                # Select the top core points for the cluster.
                # -------------------------------
                core_points_df = cluster_df[
                    (cluster_df["cluster_id"] == cluster) & (cluster_df["core_point"])
                ]
                core_points_df = core_points_df.sort_values(
                    by="membership_strength", ascending=False
                )
                core_points_texts = (
                    core_points_df[self.content_col_name].head(self.core_top_n).tolist()
                )

                # -------------------------------
                # Retrieve stratified peripheral texts to assist in updating the label.
                # -------------------------------
                peripheral_points_texts = self.get_peripheral_points(
                    cluster_df, cluster
                )

                # -------------------------------
                # Prepare centroid texts from all other clusters for context.
                # -------------------------------
                other_clusters = [c for c in clusters if c != cluster and c != -1]
                other_centroids_texts = [
                    self.get_centroid_text(cluster_df, c) for c in other_clusters
                ]

                # -------------------------------
                # Schedule the asynchronous call to update the topic label.
                # -------------------------------
                task = asyncio.create_task(
                    self.generalized_label(
                        core_points=core_points_texts,
                        core_label=core_label,
                        peripheral_points=peripheral_points_texts,
                        other_centroids=other_centroids_texts,
                    )
                )
                tasks.append((cluster, task))

            # -------------------------------
            # Await completion of all topic update tasks.
            # -------------------------------
            self.logger.debug("Waiting for all topic update tasks to complete.")
            results = await asyncio.gather(
                *(t for _, t in tasks), return_exceptions=True
            )

            # -------------------------------
            # Check for any exceptions during processing.
            # -------------------------------
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error("Error during updating topics: %s", str(result))
                    raise result

            # -------------------------------
            # Build a mapping of cluster IDs to their updated topic labels.
            # -------------------------------
            updated_topics = {
                cluster: label for (cluster, _), label in zip(tasks, results)
            }
            self.logger.debug("Updated topics generated: %s", updated_topics)
            return updated_topics

        except Exception as e:
            self.logger.error("An error occurred in update_topics_async: %s", str(e))
            raise

    def add_labels_to_cluster_df(
        self, clustered_df: pd.DataFrame, labels: dict
    ) -> pd.DataFrame:
        """
        Add topic labels to the cluster DataFrame.

        This method integrates the topic labels into the original DataFrame by adding a new 'topic' column.
        For noise points (where cluster_id == -1), the label 'Noise' is assigned.
        For clusters without an assigned label, a default label in the format 'Unlabeled_{cluster}' is used.

        Args:
            clustered_df (pd.DataFrame): DataFrame containing cluster data with a 'cluster_id' column.
            labels (dict): Dictionary mapping cluster IDs to topic labels.

        Returns:
            pd.DataFrame: A new DataFrame with an added 'topic' column containing the semantic labels.
        """
        self.logger.info("Adding labels to cluster DataFrame...")

        # -------------------------------
        # Create a copy of the original DataFrame to avoid side effects.
        # -------------------------------
        labeled_clusters_df = clustered_df.copy()
        labeled_clusters_df["cluster_id"] = pd.to_numeric(
            labeled_clusters_df["cluster_id"]
        )

        # -------------------------------
        # For each row, assign the label based on cluster_id:
        # - 'Noise' for noise points (cluster_id == -1)
        # - A mapped label or default string for clusters.
        # -------------------------------
        labeled_clusters_df["topic"] = labeled_clusters_df["cluster_id"].apply(
            lambda x: "Noise" if x == -1 else labels.get(x, f"Unlabeled_{x}")
        )
        self.logger.info("Unique topics: %d", labeled_clusters_df["topic"].nunique())
        self.logger.info(
            "\n%s",
            labeled_clusters_df[["cluster_id", "topic"]]
            .drop_duplicates()
            .sort_values("cluster_id"),
        )
        return labeled_clusters_df


def validate_labeling_params(
    cluster_df,
    llm_model,
    language,
    temperature,
    data_description,
    ascending,
    core_top_n,
    peripheral_n,
    num_strata,
    content_col_name,
):
    """
    Validate that all provided parameters for add_labels are of the expected types
    and within acceptable ranges. Raises a ValueError with an informative message if any
    parameter is invalid.
    Additionally:
      - Ensures that cluster_df is not empty.
      - Verifies that the DataFrame contains the column specified by content_col_name,
        or if not, that it contains a 'content' column.
    """
    import pandas as pd

    # Validate cluster_df:
    if not isinstance(cluster_df, pd.DataFrame):
        raise ValueError("cluster_df must be a pandas DataFrame")
    if cluster_df.empty:
        raise ValueError("cluster_df is empty")

    # Validate LLM model identifier:
    if not isinstance(llm_model, str) or not llm_model.strip():
        raise ValueError("llm_model must be a non-empty string")

    # Validate language:
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")

    # Validate temperature:
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("temperature must be a non-negative number")

    # Validate data_description:
    if not isinstance(data_description, str):
        raise ValueError("data_description must be a string")

    # Validate ascending flag:
    if not isinstance(ascending, bool):
        raise ValueError("ascending must be a boolean")

    # Validate core_top_n:
    if not isinstance(core_top_n, int) or core_top_n < 1:
        raise ValueError("core_top_n must be an integer greater than or equal to 1")

    # Validate peripheral_n:
    if not isinstance(peripheral_n, int) or peripheral_n < 1:
        raise ValueError("peripheral_n must be an integer greater than or equal to 1")

    # Validate num_strata:
    if not isinstance(num_strata, int) or num_strata < 1:
        raise ValueError("num_strata must be an integer greater than or equal to 1")

    # Validate content_col_name:
    if not isinstance(content_col_name, str) or not content_col_name.strip():
        raise ValueError("content_col_name must be a non-empty string")

    # Ensure the content column exists:
    if content_col_name not in cluster_df.columns:
        if "content" not in cluster_df.columns:
            raise ValueError(
                f"cluster_df must contain a column named '{content_col_name}' or 'content'"
            )

    return True


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
    Functional interface for labeling clusters with a user-friendly API.

    This function accepts a range of keyword arguments to configure the labeling process,
    without requiring the user to instantiate configuration objects. Internally, it creates
    configuration objects and then processes the cluster DataFrame to assign semantic topic labels.

    Args:
        cluster_df (pd.DataFrame): DataFrame containing clustering results.
        llm_model (str): Identifier for the language model (default: 'o1').
        language (str): The language for output labels (default: 'english').
        temperature (float): Temperature parameter for the language model (default: 1.0).
        data_description (str): Data context to be appended to prompts.
        ascending (bool): Order for processing clusters (default: False).
        core_top_n (int): Number of top core points to consider (default: 10).
        peripheral_n (int): Number of peripheral points to sample (default: 12).
        num_strata (int): Number of strata for sampling peripheral points (default: 3).
        content_col_name (str): Column name in cluster_df that contains text content (default: 'content').

    Returns:
        pd.DataFrame: The input DataFrame augmented with a new 'topic' column containing semantic labels.
    """
    try:
        # Validate parameters first.
        validate_labeling_params(
            cluster_df,
            llm_model,
            language,
            temperature,
            data_description,
            ascending,
            core_top_n,
            peripheral_n,
            num_strata,
            content_col_name,
        )

        # Build configuration objects internally from the provided keyword arguments.
        llm_settings = LLMSettings(
            model=llm_model,
            language=language,
            temperature=temperature,
            data_description=data_description,
        )
        sampling_settings = SamplingSettings(
            ascending=ascending,
            core_top_n=core_top_n,
            peripheral_n=peripheral_n,
            num_strata=num_strata,
            content_col_name=content_col_name,
        )

        # Instantiate the LabelingEngine with the internal configuration.
        labeler = LabelingEngine(
            llm_settings=llm_settings, sampling_settings=sampling_settings
        )

        # Generate the initial topics and then update them asynchronously.
        initial_topics = asyncio.run(labeler.generate_initial_topics_async(cluster_df))
        updated_topics = asyncio.run(
            labeler.update_topics_async(cluster_df, initial_topics)
        )

        # Add the generated labels to the original DataFrame.
        labeled_df = labeler.add_labels_to_cluster_df(
            clustered_df=cluster_df, labels=updated_topics
        )

        # Create the mapping from cluster_id to topic.
        label_dict = (
            labeled_df[["cluster_id", "topic"]]
            .drop_duplicates(subset=["cluster_id", "topic"])
            .set_index("cluster_id")["topic"]
            .to_dict()
        )

        # Return both outputs in a dictionary.
        return {"dataframe": labeled_df, "labels_dict": label_dict}

    except Exception as e:
        logging.error("An error occurred during labeling: %s", str(e))
        raise
