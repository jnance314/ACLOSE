import pandas as pd
import json
import asyncio
from openai import OpenAI, AsyncOpenAI
import os

#--------
# Placeholder for global (class) variables
#--------
llm_model = 'o1'
language = 'english'
ascending = False
core_top_n = 12
peripheral_n = 12
trace = True
#--------


#& OpenAI
OPENAI_API_KEY = os.getenv(key="OPENAI_API_KEY")
HELICONE_API_KEY = os.getenv(key="HELICONE_API_KEY")
oai_client = OpenAI(api_key=OPENAI_API_KEY, base_url="http://oai.hconeai.com/v1")
async_oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url="http://oai.hconeai.com/v1")

#--------
# Labeling
#--------
def make_model_args(llm_model, system_prompt, core_points, other_centroids, trace):
    """
    Assemble the model arguments for the given LLM model and trace setting.
    """
    model_args = {
        "model": llm_model,
        "temperature": 1.0,
        "messages": [
            {"role": "user", "content": f"""
            {system_prompt}
            Core texts from target cluster:\n{core_points} \n\nCentroid texts from other clusters:\n{other_centroids}
            """}
        ]
    }
    if trace:
        model_args["extra_headers"] = {
            "Helicone-Auth": f"Bearer {HELICONE_API_KEY}",
            "Helicone-Retry-Enabled": "true",
            "Helicone-Property-Function": "topic-labeling",
        }
    return model_args
    
async def assign_topic_to_core_points(
    core_points: list[str], 
    other_centroids: list[str],
    data_description: str
    ):
    """
    Assign a topic label to a cluster based on the top core points.
    """
    print("\nTopic labeling...\n")
    json_reminder=f"""
    Your response should always begin with
    ```json
    {{
        "step_1_target_cluster_themes":
    """
    
    system_prompt = f"""
        ## Task:
        - A large corpus of text excerpts has been embedded using transformer-based language models. The vectors have gone through dimensionality reduction and clustering, and we are now trying to assign a topic to each cluster.
        - A collection of core points with high membership scores from a single cluster is provided, along with the texts corresponding to the centroids of the other clusters. The end goal is to assign a topic label to the cluster in question that best represents the cluster, while at the same time differentiating it from the other clusters' centroid texts.
        - However, we will not simply assign a label in one step. Instead, we will reason in steps, finally arriving at a label after thinking out loud a couple times.
        - You will respond as valid JSON in a schema described below. DO NOT BE CONVERSATIONAL IN YOUR RESPONSE. Instead, respond only as a single JSON object as described in the schema.
        
        {data_description}
        
        ## What makes a good topic label?
        - A good topic label should be specific enough to differentiate the cluster from others, but general enough to encompass the core points in the cluster.
        - It should be a noun or noun phrase that describes the main theme or topic of the messages in the cluster.
        - It should not be too specific or too general- we want to address the classic problem in machine learning of bias-variance tradeoff. The label should fit the examples well, while also being general enough to apply to new examples (the other points in the cluster that we did not provide).
        - To aid you in addressing the specificity-generalization tradeoff, we have provided the texts corresponding to the centroids of the other clusters. You should consider these texts when proposing a label. By definition, other clusters are more dissimilar to the cluster in question, so the label you propose should be specific enough to differentiate the cluster in question from these other clusters' centroids.
        
        ## Five-step process:
        - Before you start, ground your thinking by taking the background context described above as a given. Then take a step back to consider all of the representative texts as a whole. Then, go through the following steps:
        
        - step 1: Think out loud about the themes or topics that you see in the representative texts of the target cluster.
        - step 2: What makes the target cluster distinct from the other clusters, given the representative texts of the target, together with the centroid texts of the other clusters? How are they similar? Think about the specificity-generalization tradeoff.
        - step 3: Think about which themes are good candidates for the target cluster, according to the above criteria describing the specificity-generalization tradeoff.
        - step 4: Propose a few labels to best represent the target cluster.
        - step 5: Finally, choose the best label that you think will represent the target cluster in the presence of other clusters.
        
        ## JSON Schema:
        - You must respond only as a JSON object with this structure:
        {{
            "step_1_target_cluster_themes": <response>,
            "step_2_other_clusters_comparison": <response>,
            "step_3_target_theme_candidates": <response>,
            "step_4_proposed_target_labels": <response>,
            "step_5_final_target_label": <label in 10 words or less>
        }}
        
        DO NOT BE CONVERSATIONAL IN YOUR RESPONSE. Instead, respond only as a single JSON object as described in the schema. 
        
        {json_reminder if llm_model in ['o1', 'o1-preview', 'o1-mini'] else None}
    """

    max_attempts = 5
    attempt_count = 0

    while attempt_count < max_attempts:
        try:
            LLMresult = await async_oai_client.chat.completions.create(**make_model_args(llm_model=llm_model, system_prompt=system_prompt, core_points=core_points, other_centroids=other_centroids, trace=trace))
            content = str(LLMresult.choices[0].message.content).replace('```','').replace('json','').strip()
            parsed_json = json.loads(content)
            label = parsed_json["step_5_final_target_label"]
            return label
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Attempt {attempt_count + 1}/{max_attempts} failed - Error processing response: {str(e)}")
            attempt_count += 1
            if attempt_count == max_attempts:
                raise Exception(f"Failed to process response after {max_attempts} attempts")
            continue
        except Exception as e:
            print(f"Attempt {attempt_count + 1}/{max_attempts} failed - Unexpected error: {str(e)}")
            attempt_count += 1
            if attempt_count == max_attempts:
                raise Exception(f"Failed to process response after {max_attempts} attempts")
            continue

async def generalized_label(
    core_points: list[str], 
    core_label: str, 
    peripheral_points: list[str], 
    other_centroids: list[str],
    data_description: str 
    ):
    """
    Generalize the label for a cluster by considering peripheral points.
    """
    print("\nGeneralizing label...\n")

    json_reminder=f"""
    Your response should always begin with
    ```json
    {{
        "step_1_target_cluster_themes":
    """
        
    system_prompt = f"""
        ## Task:
        - A large corpus of text excerpts has been embedded using transformer-based language models. The vectors have gone through dimensionality reduction and clustering, and we are now trying to assign a topic to each cluster.
        - A collection of core points with high membership scores from a single cluster is provided, along with the proposed label for the entire cluster that was produced from a process where only some core points were visible. Also given are texts corresponding to the centroids of the other clusters. In this task, you are also given a sample of peripheral texts from the target cluster, which you will use to update the original label so that it retains specificity while generalizing to a sample of points that are more peripheral to the cluster.
        - The end goal is to assign a topic label that best represents the whole target cluster, while at the same time differentiating it from the other clusters' centroid texts.
        - However, we will not simply assign a label in one step. Instead, we will reason in steps, finally arriving at a label after thinking out loud a couple times.
        - You will respond as valid JSON in a schema described below. DO NOT BE CONVERSATIONAL IN YOUR RESPONSE. Instead, respond only as a single JSON object as described in the schema.
        {f"- The final labels are going to be read by Taiwanese people, so you must write your step_5_final_target_label in Traditional {language} (繁體字)." if language != 'english' else ""}
        
        {data_description}
        
        ## What makes a good topic label?
        - A good topic label should be specific enough to differentiate the cluster from others, but general enough to encompass the core points in the cluster.
        - It should be a noun or noun phrase that describes the main theme or topic of the messages in the cluster.
        - It should not be too specific or too general- we want to address the classic problem in machine learning of bias-variance tradeoff. The label should fit the examples well, while also being general enough to apply to new examples (the other points in the cluster that we did not provide).
        - To aid you in addressing the specificity-generalization tradeoff, we have provided the texts corresponding to the centroids of the other clusters. You should consider these texts when proposing a label. By definition, other clusters are more dissimilar to the cluster in question, so the label you propose should be specific enough to differentiate the cluster in question from these other clusters' centroids.
        
        ## Five-step process:
        The task is to update the original label that was assigned, given visibility to new points in the cluster. So start with the original label and then update it based on the sampled peripheral texts. Then, go through the following steps:

        - step 1: Think out loud about the themes or topics that you see in the core texts of the target cluster, together with the original label. How do the peripheral texts relate to these themes? To what degree does the label encompass the peripheral texts as well as the core texts?
        - step 2: Think about how the label might need to be updated (if at all), given the target cluster's peripheral texts.
        - step 3: What makes the target cluster distinct from the other clusters, given the representative texts (core + peripheral) of the target, together with the centroid texts of the other clusters? How are they similar? Think about the specificity-generalization tradeoff.
        - step 4: Propose a few labels to best represent the target cluster, given the new information. It is okay if the final label is the same as the original label, but you should justify why it is still the best label.
        - step 5: Finally, choose the best label that you think will represent the target cluster in the presence of other clusters, given the sample of peripheral points. 
        
        ## JSON Schema:
        - You must respond only as a JSON object with this structure:
        {{
            "step_1_target_cluster_themes": <response>,
            "step_2_label_update_consideration": <response>,
            "step_3_other_clusters_comparison": <response>,
            "step_4_proposed_target_labels": <response>,
            "step_5_final_target_label": <label in 10 words or less>
        }}
        {f"- REMINDER: The final labels are going to be read by Taiwanese people, so you must write your step_5_final_target_label in Traditional {language} (繁體字)." if language != 'english' else ""}
        DO NOT BE CONVERSATIONAL IN YOUR RESPONSE. Instead, respond only as a single JSON object as described in the schema.
        
        {json_reminder if llm_model in ['o1', 'o1-preview', 'o1-mini'] else None}
    """
    max_attempts = 5
    attempt_count = 0

    while attempt_count < max_attempts:
        try:
            LLMresult = await async_oai_client.chat.completions.create(**make_model_args(llm_model=llm_model, system_prompt=system_prompt, core_points=core_points, other_centroids=other_centroids, trace=trace))
            content = str(LLMresult.choices[0].message.content).replace('```','').replace('json','').strip()
            parsed_json = json.loads(content)
            label = parsed_json["step_5_final_target_label"]
            return label
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Attempt {attempt_count + 1}/{max_attempts} failed - Error processing response: {str(e)}")
            attempt_count += 1
            if attempt_count == max_attempts:
                raise Exception(f"Failed to process response after {max_attempts} attempts")
            continue
        except Exception as e:
            print(f"Attempt {attempt_count + 1}/{max_attempts} failed - Unexpected error: {str(e)}")
            attempt_count += 1
            if attempt_count == max_attempts:
                raise Exception(f"Failed to process response after {max_attempts} attempts")
            continue

async def generate_initial_topics_async(cluster_df: pd.DataFrame):
    """
    Generate initial topics for each cluster using the top N core points and centroids of other clusters.
    """    
    # Get clusters and their sizes, excluding outliers (-1)
    cluster_sizes = cluster_df[cluster_df['cluster_number'] != -1]['cluster_number'].value_counts()

    # Sort clusters by size
    clusters = cluster_sizes.index.tolist() if ascending else cluster_sizes.index.tolist()[::-1]
    
    initial_topics = {}
    tasks = []

    for cluster in clusters:
        if cluster == -1:
            continue  # Skip outliers

        # Get top N core points for the cluster
        core_points_df = cluster_df[
            (cluster_df['cluster_number'] == cluster) & (cluster_df['core_point'])
        ]
        core_points_df = core_points_df.sort_values(by='membership_strength', ascending=False)
        core_points_texts = core_points_df['content'].head(core_top_n).tolist()

        # Get centroids (point closest to cluster center) of other clusters
        other_clusters = [c for c in clusters if c != cluster and c != -1]
        other_centroids_texts = [
            get_centroid_text(cluster_df=cluster_df, cluster=c) for c in other_clusters
        ]

        # Prepare the task
        task = asyncio.ensure_future(
            assign_topic_to_core_points(
                core_points=core_points_texts,
                other_centroids=other_centroids_texts, # type: ignore
            )
        ) # type: ignore
        tasks.append((cluster, task))

    # Run all tasks concurrently
    results = await asyncio.gather(*(task for cluster, task in tasks))

    # Collect results
    initial_topics = {cluster: label for (cluster, task), label in zip(tasks, results)}

    return initial_topics

def get_peripheral_points(
    cluster_df: pd.DataFrame, 
    cluster: int, 
    num_strata: int
    ):
    """
    Get stratified peripheral points for a given cluster.

    Parameters:
    - cluster_df: The DataFrame containing clustering information.
    - cluster: The cluster number for which to get peripheral points.
    - choice: The column name to select ('user_query' or 'bot_answer').
    - peripheral_n: The total number of peripheral points to sample.
    - num_strata: The number of strata for stratification.

    Returns:
    - peripheral_points_texts: A list of peripheral texts.
    """
    peripheral_points_df = cluster_df[(cluster_df['cluster_number'] == cluster) & (~cluster_df['core_point'])]

    if peripheral_points_df.empty:
        return []
    
    # For HDBSCAN, stratify based on inverse membership strength
    peripheral_points_df = peripheral_points_df.copy()
    peripheral_points_df['stratum'] = pd.qcut(
        -peripheral_points_df['membership_strength'],
        q=min(num_strata, len(peripheral_points_df)),
        labels=False,
        duplicates='drop'
    )

    # Sample from each stratum
    peripheral_points_texts = []
    for stratum in peripheral_points_df['stratum'].unique():
        stratum_df = peripheral_points_df[peripheral_points_df['stratum'] == stratum]
        n_samples_per_stratum = max(1, peripheral_n // num_strata)
        sampled_texts = stratum_df['content'].sample(
            n=min(n_samples_per_stratum, len(stratum_df)),
            random_state=42,
            replace=False
        ).tolist()
        peripheral_points_texts.extend(sampled_texts)

    # Ensure we have at least peripheral_n samples
    if len(peripheral_points_texts) < peripheral_n:
        additional_needed = peripheral_n - len(peripheral_points_texts)
        remaining_points = peripheral_points_df[~peripheral_points_df['content'].isin(peripheral_points_texts)]
        if not remaining_points.empty:
            additional_texts = remaining_points['content'].sample(
                n=min(additional_needed, len(remaining_points)),
                random_state=42,
                replace=False
            ).tolist()
            peripheral_points_texts.extend(additional_texts)

    return peripheral_points_texts

def get_centroid_text(
    cluster_df: pd.DataFrame, 
    cluster: int
    ):
    """
    Get the centroid text for a given cluster.

    Parameters:
    - cluster_df: The DataFrame containing clustering information.
    - cluster: The cluster number.
    - choice: The column name to select ('user_query' or 'bot_answer').

    Returns:
    - centroid_text: The text corresponding to the cluster centroid.
    """
    cluster_data = cluster_df[cluster_df['cluster_number'] == cluster]
    
    # For HDBSCAN, use the top core point
    core_points_df = cluster_data[cluster_data['core_point']]
    core_points_df = core_points_df.sort_values(by='membership_strength', ascending=False)
    if not core_points_df.empty:
        centroid_point = core_points_df.iloc[0]
    else:
        centroid_point = cluster_data.iloc[0]
        
    centroid_text = centroid_point['content']
    return centroid_text

async def update_topics_async( cluster_df: pd.DataFrame, initial_topics: dict, num_strata: int = 3):
    """
    Update topics for each cluster using stratified sampling of peripheral points and the generalized_label function.
    """    
    # Get clusters and their sizes, excluding outliers (-1)
    cluster_sizes = cluster_df[cluster_df['cluster_number'] != -1]['cluster_number'].value_counts()
    
    # Sort clusters by size
    clusters = cluster_sizes.index.tolist() if ascending else cluster_sizes.index.tolist()[::-1]
    
    updated_topics = {}
    tasks = []

    for cluster in clusters:
        if cluster == -1:
            continue  # Skip outliers

        core_label = initial_topics.get(cluster, "Unknown")

        # Get core points
        core_points_df = cluster_df[(cluster_df['cluster_number'] == cluster) & (cluster_df['core_point'])]
        core_points_df = core_points_df.sort_values(by='membership_strength', ascending=False)
        core_points_texts = core_points_df['content'].head(12).tolist()

        # Get peripheral points using the new function
        peripheral_points_texts = get_peripheral_points(
            cluster_df=cluster_df,
            cluster=cluster,
            num_strata=num_strata
        )

        # Get centroids (point closest to cluster center) of other clusters
        other_clusters = [c for c in clusters if c != cluster and c != -1]
        other_centroids_texts = [
            get_centroid_text(cluster_df, c) for c in other_clusters
        ]

        # Prepare the task
        task = asyncio.ensure_future(
            generalized_label(
                core_points=core_points_texts,
                core_label=core_label,
                peripheral_points=peripheral_points_texts,
                other_centroids=other_centroids_texts, # type: ignore
            )
        ) # type: ignore
        tasks.append((cluster, task))

    # Run all tasks concurrently
    results = await asyncio.gather(*(task for cluster, task in tasks))

    # Collect results
    updated_topics = {cluster: label for (cluster, task), label in zip(tasks, results)}

    return updated_topics

def add_labels_to_cluster_df(
    clustered_df: pd.DataFrame, 
    labels: dict
    ):
    """
    Add semantic topic labels to the cluster DataFrame.
    Handles noise cluster (cluster_number = -1) explicitly.
    """
    print("\nAdding labels to cluster DataFrame...\n")
    
    # Create a copy to avoid modifying the original DataFrame
    labeled_clusters_df = clustered_df.copy()
    
    # Convert cluster_number to numeric if it's not already
    labeled_clusters_df['cluster_number'] = pd.to_numeric(labeled_clusters_df['cluster_number'])
    
    # Handle noise cluster separately
    labeled_clusters_df['topic'] = labeled_clusters_df['cluster_number'].apply(
        lambda x: 'Noise' if x == -1 else labels.get(x, f'Unlabeled_{x}')
    )
    
    print(f"Unique topics: {labeled_clusters_df['topic'].nunique()}")
    # print the unique cluster number and its corresponding topic, each pair on a new line
    print(labeled_clusters_df[['cluster_number', 'topic']].drop_duplicates().sort_values('cluster_number'))
    return labeled_clusters_df