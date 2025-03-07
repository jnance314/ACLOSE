import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import re
import plotly.graph_objects as go
import plotly.express as px
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _color_map(df: pd.DataFrame, color_palette=px.colors.qualitative.Alphabet) -> dict:
    """
    Create a mapping from each cluster ID to a color from the specified palette.

    Noise clusters (with cluster_id == -1) are excluded from the mapping.
    When a cluster is not found in the mapping (e.g., noise), a default color (gray) is used.

    Args:
        - df (pd.DataFrame): DataFrame containing a 'cluster_id' column.
        - color_palette (list, optional): List of color strings to use. Defaults to px.colors.qualitative.Alphabet.

    Returns:
        - dict: Dictionary mapping valid cluster IDs (int) to color strings.
    """
    logger.debug("Starting _color_map.")
    # -------------------------------
    # Validate input DataFrame contains required column
    # -------------------------------
    if "cluster_id" not in df.columns:
        logger.error("DataFrame does not contain 'cluster_id' column.")
        raise ValueError("DataFrame must contain 'cluster_id' column.")

    logger.info(f"Creating cluster color map using {len(color_palette)} colors...")
    # -------------------------------
    # Identify unique clusters (excluding noise labeled as -1)
    # -------------------------------
    unique_clusters = sorted([c for c in df["cluster_id"].unique() if c != -1])
    logger.debug(f"Unique clusters (excluding noise): {unique_clusters}")

    # -------------------------------
    # Build a color map: assign a color to each cluster using the palette, cycling if necessary
    # -------------------------------
    color_map = {}
    if len(unique_clusters) > len(color_palette):
        for i, c in enumerate(unique_clusters):
            color_map[c] = color_palette[i % len(color_palette)]
    else:
        for i, c in enumerate(unique_clusters):
            color_map[c] = color_palette[i]

    logger.debug(f"Color map created: {color_map}")
    return color_map


def _prepare_topic_counts(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame of topic counts from clustering results.

    The function filters out noise (cluster_id == -1). If a 'topic' column is present,
    counts are grouped by both 'cluster_id' and 'topic'; otherwise, a default topic label is created.

    Args:
        - clusters_df (pd.DataFrame): DataFrame with clustering results.

    Returns:
        - pd.DataFrame: Processed DataFrame with columns for 'cluster_id', 'topic', and 'count',
                      sorted in ascending order by count.
    """
    logger.debug("Starting _prepare_topic_counts.")
    # -------------------------------
    # Validate DataFrame contains the required 'cluster_id' column
    # -------------------------------
    if "cluster_id" not in clusters_df.columns:
        logger.error("DataFrame does not contain 'cluster_id' column.")
        raise ValueError("DataFrame must contain 'cluster_id' column.")

    # -------------------------------
    # Filter out noise cluster (-1) from the data
    # -------------------------------
    topic_df = clusters_df[clusters_df["cluster_id"] != -1].copy()
    logger.debug(f"Filtered DataFrame shape (excluding noise): {topic_df.shape}")

    # -------------------------------
    # Group the data to count topics per cluster, handling missing 'topic' column
    # -------------------------------
    if "topic" in topic_df.columns:
        topic_counts = (
            topic_df.groupby(["cluster_id", "topic"]).size().reset_index(name="count")
        )
    else:
        topic_counts = topic_df.groupby("cluster_id").size().reset_index(name="count")
        # When 'topic' information is not provided, create a default label for each cluster
        topic_counts["topic"] = topic_counts["cluster_id"].apply(
            lambda c: f"Cluster {c}"
        )

    # -------------------------------
    # Sort counts for presentation; ascending order can be adjusted as needed
    # -------------------------------
    topic_counts = topic_counts.sort_values("count", ascending=True)
    logger.debug("Topic counts prepared.")
    return topic_counts


def _cluster_bar_chart_fig(clusters_df: pd.DataFrame, color_map: dict) -> go.Figure:
    """
    Make a horizontal bar chart of cluster sizes with topic labels.

    The function uses the processed topic counts to generate a horizontal bar chart.
    Bars are colored according to the provided color map, with default gray for missing entries.

    Args:
        - clusters_df (pd.DataFrame): DataFrame with clustering results. Must contain 'cluster_id' and, optionally, 'topic'.
        - color_map (dict): Dictionary mapping valid cluster IDs to color strings.

    Returns:
        - go.Figure: Plotly Figure object representing the bar chart.
    """
    logger.debug("Generating cluster bar chart.")
    # -------------------------------
    # Prepare data for plotting using topic counts
    # -------------------------------
    topic_counts = _prepare_topic_counts(clusters_df)
    fig = go.Figure()

    # -------------------------------
    # Determine bar colors based on cluster_id using provided color map; default to gray if not found
    # -------------------------------
    bar_colors = [
        color_map.get(row.cluster_id, "gray") for _, row in topic_counts.iterrows()
    ]
    logger.debug(f"Bar colors assigned: {bar_colors}")

    # -------------------------------
    # Add horizontal bar trace to the figure with counts as lengths
    # -------------------------------
    fig.add_trace(
        go.Bar(
            y=list(range(len(topic_counts))),
            x=topic_counts["count"],
            orientation="h",
            marker=dict(
                color=bar_colors, line=dict(color="rgba(0,0,0,0.3)", width=0.5)
            ),
            text=topic_counts["count"],
            textposition="outside",
            hoverinfo="skip",
        )
    )

    # -------------------------------
    # Add cluster topic labels as text annotations next to the bars
    # -------------------------------
    fig.add_trace(
        go.Scatter(
            x=[0] * len(topic_counts),
            y=[i + 0.5 for i in range(len(topic_counts))],
            mode="text",
            text=[
                f"Cluster {row.cluster_id}: {row.topic}"
                for _, row in topic_counts.iterrows()
            ],
            textposition="middle right",
            textfont=dict(size=12, color="white"),
            showlegend=False,
        )
    )

    # -------------------------------
    # Update layout settings for a dark theme and improved visualization
    # -------------------------------
    fig.update_layout(
        title=dict(
            text="Topics by Cluster Size", x=0.5, font=dict(size=24, color="white")
        ),
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor="black",
        paper_bgcolor="black",
        width=900,
        bargroupgap=0.5,
        height=max(400, len(topic_counts) * 50),
        margin=dict(l=0, r=40, t=100, b=80),
        showlegend=False,
        yaxis=dict(showticklabels=False, range=[-0.5, len(topic_counts)]),
        font=dict(size=14),
    )

    # -------------------------------
    # Configure x-axis grid and zero-line for better readability
    # -------------------------------
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.1)",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="rgba(0,0,0,0.2)",
    )

    logger.debug("Cluster bar chart created successfully.")
    return fig


def _silhouette_fig(clustered_df: pd.DataFrame, color_map: dict) -> go.Figure:
    """
    Generate an enhanced silhouette plot using Plotly, ordered by cluster size.

    The function computes silhouette scores (using Euclidean distance) for the clustering results
    (ignoring noise clusters) and visualizes them. A vertical dashed line indicates the average silhouette score.

    Args:
        - clustered_df (pd.DataFrame): DataFrame containing clustering results with 'reduced_vector' and 'cluster_id'.
        - color_map (dict): Dictionary mapping valid cluster IDs to color strings.

    Returns:
        - go.Figure: Plotly Figure object representing the silhouette plot.

    Raises:
        - ValueError: If 'reduced_vector' is missing or if there are fewer than 2 valid clusters (excluding noise).
    """
    logger.debug("Starting silhouette plot generation.")
    # -------------------------------
    # Validate presence of required 'reduced_vector' column in the DataFrame
    # -------------------------------
    if "reduced_vector" not in clustered_df.columns:
        logger.error("DataFrame does not contain 'reduced_vector' column.")
        raise ValueError("No 'reduced_vector' column found in DataFrame")

    # -------------------------------
    # Attempt to convert 'reduced_vector' column to numpy array for processing
    # -------------------------------
    try:
        data = np.vstack(clustered_df["reduced_vector"].values)  # type: ignore
    except Exception as e:
        logger.error("Error converting 'reduced_vector' column to numpy array.")
        raise ValueError("Error processing 'reduced_vector' column.") from e

    # -------------------------------
    # Filter out noise clusters (labelled as -1) for silhouette analysis
    # -------------------------------
    clusters = clustered_df["cluster_id"].values
    mask = clusters != -1
    data_filtered = data[mask]
    clusters_filtered = clusters[mask]

    # -------------------------------
    # Check that there are at least 2 valid clusters to compute silhouette scores
    # -------------------------------
    if len(np.unique(clusters_filtered)) < 2:
        logger.error("Not enough valid clusters to compute silhouette scores.")
        raise ValueError("Need at least 2 valid clusters to create silhouette plot")

    # -------------------------------
    # Compute silhouette values and average silhouette score using Euclidean distance
    # -------------------------------
    silhouette_vals = silhouette_samples(
        data_filtered, clusters_filtered, metric="euclidean"
    )
    silhouette_avg = silhouette_score(
        data_filtered, clusters_filtered, metric="euclidean"
    )
    logger.debug(f"Silhouette average score: {silhouette_avg:.3f}")

    # -------------------------------
    # Initialize Plotly figure and prepare variables for layout spacing
    # -------------------------------
    fig = go.Figure()
    cluster_sizes = {
        c: np.sum(clusters_filtered == c) for c in np.unique(clusters_filtered)
    }
    sorted_clusters = sorted(
        cluster_sizes.keys(), key=lambda c: cluster_sizes[c], reverse=True
    )

    spacing = 10
    total_height = 10  # initial offset
    # Calculate total height required for the plot based on cluster sizes and spacing
    for c in sorted_clusters:
        total_height += cluster_sizes[c] + spacing

    y_upper = total_height

    # -------------------------------
    # Add silhouette traces for each cluster using a helper function
    # -------------------------------
    for cluster in sorted_clusters:
        y_upper = _add_silhouette_cluster(
            fig,
            cluster,
            silhouette_vals,
            clusters_filtered,
            y_upper,
            spacing,
            color_map,
        )

    # -------------------------------
    # Add a vertical line representing the average silhouette score
    # -------------------------------
    fig.add_shape(
        type="line",
        x0=silhouette_avg,
        x1=silhouette_avg,
        y0=0,
        y1=total_height,
        line=dict(color="red", dash="dash"),
    )

    # -------------------------------
    # Annotate the average silhouette score on the plot
    # -------------------------------
    fig.add_annotation(
        x=silhouette_avg,
        y=total_height,
        text=f"Average silhouette: {silhouette_avg:.3f}",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="white"),
    )

    # -------------------------------
    # Update layout settings for the silhouette plot with dark theme and grid lines
    # -------------------------------
    fig.update_layout(
        title=dict(text="Silhouette Plot", x=0.5, font=dict(size=24, color="white")),
        xaxis_title="Silhouette Coefficient",
        xaxis=dict(
            range=[-0.2, 1],
            showgrid=True,
            gridcolor="gray",
            zeroline=True,
            zerolinecolor="gray",
            color="white",
        ),
        yaxis=dict(
            range=[0, total_height],
            showticklabels=False,
            title=None,
            color="white",
            gridcolor="gray",
            zerolinecolor="gray",
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        width=900,
        height=max(500, total_height),
        showlegend=False,
        margin=dict(l=80, r=30, t=100, b=80),
    )

    logger.debug("Silhouette plot created successfully.")
    return fig


def _add_silhouette_cluster(
    fig, cluster, silhouette_vals, clusters_filtered, y_upper, spacing, color_map
):
    """
    Helper function to add silhouette traces and annotations for a single cluster.

    Args:
        - fig (go.Figure): Plotly Figure to which the cluster silhouette will be added.
        - cluster (int): The cluster ID.
        - silhouette_vals (np.array): Array of silhouette values.
        - clusters_filtered (np.array): Array of filtered cluster IDs.
        - y_upper (float): Upper y-bound for plotting.
        - spacing (int): Spacing between clusters.
        - color_map (dict): Dictionary mapping valid cluster IDs to color strings.

    Returns:
        - float: Updated y_upper value for the next cluster.
    """
    # -------------------------------
    # Extract silhouette values for the specific cluster and sort them
    # -------------------------------
    vals = silhouette_vals[clusters_filtered == cluster]
    vals.sort()
    size = len(vals)
    logger.debug(f"Adding silhouette for cluster {cluster} with {size} points.")
    y_lower = y_upper - size

    # -------------------------------
    # Add a filled area trace for the cluster silhouette
    # -------------------------------
    fillcolor = color_map.get(cluster, "gray")
    fig.add_trace(
        go.Scatter(
            x=vals,
            y=np.arange(y_lower, y_upper),
            fill="tozerox",
            mode="none",
            name=f"Cluster {cluster} (n={size})",
            fillcolor=fillcolor,
            opacity=0.7,
            hoverinfo="skip",
        )
    )

    # -------------------------------
    # Add a text annotation at the midpoint of the cluster's silhouette area
    # -------------------------------
    y_mid = (y_lower + y_upper) / 2
    fig.add_annotation(
        x=0,
        y=y_mid,
        text=f"Cluster {cluster}",
        showarrow=False,
        xanchor="left",
        font=dict(color="white", size=14),
        xshift=5,
    )

    # Return updated y_upper position for the next cluster silhouette
    return y_lower - spacing


def _wrap_text(text: str, width: int | None = None) -> str:
    """
    Helper function to wrap text at word boundaries, handling both English and Chinese.

    If width is not provided, a default width of 100 characters is used.

    Args:
        - text (str): The text to be wrapped.
        - width (int, optional): Maximum number of characters per line. Defaults to 100 if not provided.

    Returns:
        - str: Wrapped text with HTML line breaks.
    """
    logger.debug("Wrapping text.")
    # -------------------------------
    # Set wrap width to default if not provided
    # -------------------------------
    if width is None:
        width = 100
    paragraphs = text.split("\n")
    wrapped_lines = []

    # -------------------------------
    # Process each paragraph separately
    # -------------------------------
    for paragraph in paragraphs:
        if paragraph.strip():
            wrapped_lines.extend(_wrap_paragraph(paragraph.strip(), width))
        else:
            # Preserve empty lines as empty strings
            wrapped_lines.append("")

    # Join wrapped lines with HTML line breaks, filtering out any empty lines
    return "<br>".join(filter(None, wrapped_lines))


def _wrap_paragraph(text: str, width: int) -> list:
    """
    Wrap a single paragraph of text.

    Depending on whether the text contains Chinese characters, the wrapping is done differently.

    Args:
        - text (str): The paragraph to be wrapped.
        - width (int): Maximum number of characters per line.

    Returns:
        - list: List of wrapped lines.
    """
    logger.debug("Wrapping a paragraph.")
    # -------------------------------
    # Check if the text contains Chinese characters
    # -------------------------------
    if not any(_is_chinese(char) for char in text):
        # Use English-specific wrapping if no Chinese characters found
        return _wrap_english_text(text, width)

    # -------------------------------
    # For mixed Chinese text, split text into chunks (either Chinese characters or non-Chinese words)
    # -------------------------------
    chunks = _split_mixed_text(text)
    lines = []
    current_line = []
    current_length = 0

    # -------------------------------
    # Build lines by accumulating chunks until the wrap width is reached
    # -------------------------------
    for chunk in chunks:
        if not chunk.strip():
            continue

        # Determine width of chunk (same calculation for Chinese or non-Chinese in this case)
        if _is_chinese(chunk[0]):
            chunk_width = len(chunk)
        else:
            chunk_width = len(chunk)

        # Check if adding the chunk would exceed the specified width
        if current_length + chunk_width <= width:
            current_line.append(chunk)
            current_length += chunk_width
        else:
            if current_line:
                # Append the current line and start a new one
                lines.append("".join(current_line).strip())
            current_line = [chunk]
            current_length = chunk_width

    if current_line:
        lines.append("".join(current_line).strip())

    return lines


def _is_chinese(char: str) -> bool:
    """
    Check if a character is Chinese.

    Args:
        - char (str): A single character.

    Returns:
        - bool: True if the character is Chinese, False otherwise.
    """
    # -------------------------------
    # Determine if the Unicode code point falls within the range for Chinese characters
    # -------------------------------
    return "\u4e00" <= char <= "\u9fff"


def _split_mixed_text(text: str) -> list:
    """
    Split text into chunks of Chinese characters or non-Chinese words.

    The regular expression splits text into:
      - Single Chinese characters,
      - One or more consecutive non-space, non-Chinese characters, or
      - Whitespace sequences.

    Args:
        - text (str): The text to be split.

    Returns:
        - list: List of text chunks.
    """
    # -------------------------------
    # Regular expression pattern explanation:
    # - [\u4e00-\u9fff]: Matches a single Chinese character.
    # - [^\s\u4e00-\u9fff]+: Matches one or more consecutive non-space and non-Chinese characters.
    # - \s+: Matches one or more whitespace characters.
    # -------------------------------
    pattern = r"[\u4e00-\u9fff]|[^\s\u4e00-\u9fff]+|\s+"
    return [chunk for chunk in re.findall(pattern, text) if chunk]


def _wrap_english_text(text: str, width: int) -> list:
    """
    Specifically handle wrapping of English text.

    Args:
        - text (str): The English text to be wrapped.
        - width (int): Maximum number of characters per line.

    Returns:
        - list: List of wrapped lines.
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    # -------------------------------
    # Iterate over each word to build lines that do not exceed the wrap width
    # -------------------------------
    for word in words:
        word_length = len(word)
        # If adding the word fits or if current line is empty, add the word to current line
        if current_length + word_length + 1 <= width or not current_line:
            current_line.append(word)
            current_length += word_length + 1
        else:
            # Append the current line to lines and start a new line with the current word
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length

    if current_line:
        lines.append(" ".join(current_line))
    return lines


# -------------------------------------------------------------------------
# Class for Scatter (Point Cloud) Plot
# -------------------------------------------------------------------------


class ScatterPlot:
    """
    Class to generate a scatter (point cloud) plot for clustering visualization using Plotly.

    This class encapsulates input validation, trace generation for each cluster (including hover text and centroid annotations),
    and layout configuration. It supports both 2D and 3D visualizations based on the dimensionality of the reduced vectors.
    """

    def __init__(
        self,
        clusters_df: pd.DataFrame,
        color_map: dict,
        id_col_name: Optional[str] = None,
        content_col_name: str = "content",
        wrap_width: int = 100,
    ):
        """
        Initialize the ScatterPlot with clustering results and plotting configuration.

        Args:
            - clusters_df (pd.DataFrame): DataFrame containing clustering results with 'cluster_id' and 'reduced_vector' (among others).
            - color_map (dict): Dictionary mapping valid cluster IDs to color strings.
            - id_col_name: Optional column name for record IDs.
            - content_col_name (str): Name of the column containing the text content.
            - wrap_width (int): Maximum width for wrapping text in hover labels.
        """
        self.clusters_df = clusters_df
        self.color_map = color_map
        self.id_col_name = id_col_name
        self.content_col_name = content_col_name
        self.wrap_width = wrap_width
        self.dim = self._validate_and_extract_vectors()

    def _validate_and_extract_vectors(self) -> int:
        """
        Validate that the DataFrame is not empty and contains a 'reduced_vector' column.
        Also check that the first entry in 'reduced_vector' is list-like and determine its dimensionality.

        Returns:
            - int: Dimensionality of the reduced vectors.

        Raises:
            - ValueError: If the DataFrame is empty, missing the 'reduced_vector' column, or if the entries are not list-like.
        """
        logger.debug("Validating input DataFrame for scatter plot.")
        if self.clusters_df.empty:
            logger.error("Input clusters DataFrame is empty.")
            raise ValueError("Input DataFrame is empty.")

        if "reduced_vector" not in self.clusters_df.columns:
            logger.error("DataFrame does not contain 'reduced_vector' column.")
            raise ValueError("DataFrame must contain 'reduced_vector' column.")

        first_vector = self.clusters_df["reduced_vector"].iloc[0]
        if not isinstance(first_vector, (list, tuple, np.ndarray)):
            logger.error("'reduced_vector' entries are not list-like.")
            raise ValueError("'reduced_vector' entries must be list-like.")

        dim = len(first_vector)
        logger.debug(f"Determined reduced vector dimensionality: {dim}")
        return dim

    def _generate_hover_text(self, row: pd.Series) -> str:
        """
        Generate hover text for a data point in the scatter plot.

        Args:
            - row (pd.Series): A row from the DataFrame representing a single data point.

        Returns:
            - str: The formatted hover text.
        """
        hover = f"Cluster: {row['cluster_id']}"
        if "topic" in row:
            hover += f"<br>Topic: {row['topic']}<br><br>"
        if self.id_col_name is not None:
            hover += f"ID: {row[self.id_col_name]}<br>"
        hover += f"Content: {_wrap_text(text=row[self.content_col_name], width=self.wrap_width)}"
        return hover

    def _create_centroid_annotation(
        self, cluster_data: pd.DataFrame, has_3d: bool, cluster_topic: str
    ) -> go.Trace:
        """
        Create a text annotation at the centroid of a non-noise cluster.

        Args:
            - cluster_data (pd.DataFrame): Data for the current cluster.
            - has_3d (bool): Flag indicating whether to plot in 3D.
            - cluster_topic (str): The representative topic label for the cluster.

        Returns:
            - go.Trace: A Plotly trace containing the centroid annotation.
        """
        center_x = cluster_data["reduced_vector"].apply(lambda v: v[0]).mean()
        center_y = cluster_data["reduced_vector"].apply(lambda v: v[1]).mean()

        if has_3d:
            center_z = cluster_data["reduced_vector"].apply(lambda v: v[2]).mean()
            annotation = go.Scatter3d(
                x=[center_x],
                y=[center_y],
                z=[center_z],
                mode="text",
                text=[cluster_topic],
                textposition="middle center",
                textfont=dict(size=14, color="white", family="Arial Black"),
                hoverinfo="skip",
                showlegend=False,
            )
        else:
            annotation = go.Scatter(
                x=[center_x],
                y=[center_y],
                mode="text",
                text=[cluster_topic],
                textposition="middle center",
                textfont=dict(size=14, color="white", family="Arial Black"),
                hoverinfo="skip",
                showlegend=False,
            )
        return annotation  # type: ignore

    def _create_cluster_trace(
        self, cluster_data: pd.DataFrame, cluster_id: int, has_3d: bool
    ) -> List[go.Trace]:
        """
        Create scatter traces for a single cluster, including hover text and (for non-noise clusters) a centroid annotation.

        Args:
            - cluster_data (pd.DataFrame): Data for the specific cluster.
            - cluster_id (int): The ID of the cluster.
            - has_3d (bool): Flag indicating whether the plot is 3D.

        Returns:
            - List[go.Trace]: List of Plotly traces for the cluster.
        """
        traces = []
        if cluster_data.empty:
            logger.debug(f"Skipping empty cluster: {cluster_id}")
            return traces

        # Determine visual properties based on whether the cluster is noise (-1) or a valid cluster.
        is_noise = cluster_id == -1
        opacity = 0.35 if is_noise else 0.75
        symbol = "circle"
        size = 2 if is_noise else 4
        legend_name = str(cluster_id) if not is_noise else "Noise"
        point_color = self.color_map.get(cluster_id, "gray")

        # Generate hover text for each point in the cluster.
        hover_text = cluster_data.apply(self._generate_hover_text, axis=1)

        # Extract coordinates from the 'reduced_vector' column.
        x = cluster_data["reduced_vector"].apply(lambda v: v[0])
        y = cluster_data["reduced_vector"].apply(lambda v: v[1])

        if has_3d:
            z = cluster_data["reduced_vector"].apply(lambda v: v[2])
            scatter_trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=size,
                    color=point_color,
                    opacity=opacity,
                    symbol=symbol,
                    line=dict(color="rgba(0,0,0,0.2)", width=0.5),
                ),
                name=legend_name,
                text=hover_text,
                hoverinfo="text",
                showlegend=True,
            )
        else:
            scatter_trace = go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=size,
                    color=point_color,
                    opacity=opacity,
                    symbol=symbol,
                    line=dict(color="rgba(0,0,0,0.2)", width=0.5),
                ),
                name=legend_name,
                text=hover_text,
                hoverinfo="text",
                showlegend=True,
            )
        traces.append(scatter_trace)

        # For non-noise clusters, add a centroid annotation with the topic label.
        if not is_noise:
            if "topic" in cluster_data.columns:
                cluster_topic = cluster_data["topic"].mode().iloc[0]
            else:
                cluster_topic = f"Cluster {cluster_id}"
            annotation_trace = self._create_centroid_annotation(
                cluster_data, has_3d, cluster_topic
            )
            traces.append(annotation_trace)

        return traces

    def _get_layout(self, has_3d: bool) -> dict:
        """
        Configure layout settings for the scatter plot, including dark theme and legend settings.

        Args:
            - has_3d (bool): Flag indicating whether the plot is 3D.

        Returns:
            - dict: Dictionary of layout settings for the Plotly figure.
        """
        layout_args = dict(
            title=dict(text="Clusters", x=0.5, font=dict(size=24, color="white")),
            plot_bgcolor="black",
            paper_bgcolor="black",
            showlegend=True,
            height=1000,
            margin=dict(l=0, r=120, t=50, b=0),
            legend=dict(x=1, y=0.5, xanchor="left", font=dict(size=12, color="white")),
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0.8)",
                font=dict(color="white", size=12),
                align="left",
                namelength=-1,
            ),
        )

        if has_3d:
            layout_args.update(
                dict(
                    scene=dict(
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        zaxis=dict(showgrid=False, zeroline=False, visible=False),
                        bgcolor="black",
                        camera=dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.5),
                        ),
                    )
                )  # type: ignore
            )
        else:
            layout_args.update(
                dict(
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                )
            )
        return layout_args

    def create_figure(self) -> go.Figure:
        """
        Generate the scatter (point cloud) plot as a Plotly Figure.

        The function determines whether to plot in 2D or 3D based on the dimensionality of the 'reduced_vector'.
        If the dimensionality is greater than 3, a placeholder figure with a warning annotation is returned.

        Returns:
            - go.Figure: Plotly Figure object representing the scatter plot.
        """
        logger.debug("Generating clusters point cloud plot.")
        dim = self.dim
        if dim > 3:
            logger.warning("Cannot plot scatter plot with dims > 3... Skipping plot")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Data has dims={dim}. Cannot plot scatter plot with dims > 3. Skipping plot.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=20, color="red"),
            )
            return fig

        has_3d = dim == 3
        logger.debug(
            f"Detected {'3D' if has_3d else '2D'} plotting based on 'reduced_vector'."
        )

        # Initialize an empty list for traces.
        all_traces = []
        # Iterate over each unique cluster and generate corresponding traces.
        for cluster_id in sorted(self.clusters_df["cluster_id"].unique()):
            cluster_data = self.clusters_df[
                self.clusters_df["cluster_id"] == cluster_id
            ]
            traces = self._create_cluster_trace(cluster_data, cluster_id, has_3d)
            all_traces.extend(traces)

        # Create the figure and add all traces.
        fig = go.Figure(data=all_traces)
        # Update layout settings.
        layout_args = self._get_layout(has_3d)
        fig.update_layout(**layout_args)  # type: ignore
        logger.debug("Clusters point cloud plot created successfully.")
        return fig


# -------------------------------------------------------------------------
# External interface functions
# -------------------------------------------------------------------------


def silhouette_fig(clustered_df: pd.DataFrame) -> go.Figure:
    """
    Generate an enhanced silhouette plot for evaluating clustering quality.

    The silhouette plot is a powerful visualization that shows how well each point lies within
    its assigned cluster. It helps identify:
    - Well-formed clusters (high silhouette values)
    - Potential misclassifications (negative or low silhouette values)
    - The overall quality of the clustering (average silhouette score)

    This function creates a visually enhanced silhouette plot with clusters ordered by size,
    color-coded for easy identification, and includes the average silhouette score as a
    reference line.

    Parameters:
        clustered_df (pd.DataFrame):
            DataFrame containing clustering results from run_clustering(). Must include columns:
            'cluster_id' (int): cluster identifiers with -1 for noise
            'reduced_vector' (list-like): reduced dimensional representations of each point,
                                          typically 2D or 3D coordinates from UMAP reduction
            Silhouette scores are computed using Euclidean distance on these reduced vectors.

    Returns:
        go.Figure:
            A Plotly Figure object containing the silhouette plot visualization. This
            interactive figure includes:

            - Individual silhouette profiles for each cluster, color-coded to match
              other visualizations from this package
            - Clusters ordered by size (largest at top) for easier interpretation
            - Text labels identifying each cluster
            - A vertical dashed red line showing the average silhouette score
            - Dark theme styling for better visibility of silhouette patterns

            The figure can be:
            - Displayed directly with fig.show()
            - Saved as an HTML file with fig.write_html("silhouette.html")
            - Converted to static images with fig.write_image("silhouette.png")
            - Customized further using Plotly's update_layout() and other methods
            - Embedded in Jupyter notebooks, dashboards, or web applications

            The silhouette values range from -1 to 1:
            - Values near 1 indicate points well-matched to their clusters
            - Values near 0 indicate points on cluster boundaries
            - Negative values suggest potential misclassifications

            When analyzing this plot, pay attention to:
            - The width of each cluster's profile (indicates cluster size)
            - The shape of each profile (consistent high values indicate coherent clusters)
            - Clusters with many negative values (may indicate fragmented clusters)
            - The average silhouette score (higher is better, >0.5 is generally good)

    Notes:
        - Noise points (cluster_id = -1) are excluded from the silhouette calculation.
        - The function automatically assigns colors to clusters for visual distinction.
        - Higher silhouette values (closer to 1.0) indicate better clustering.
        - Average silhouette scores above 0.5 generally indicate reasonable clustering;
          scores above 0.7 indicate strong clustering structure.
        - The plot arranges clusters by size (largest first) to make the visualization
          more interpretable.
        - Clusters with many points near or below 0 may benefit from re-clustering or
          parameter adjustments.

    Example interpretation:
        - If most clusters show high silhouette values (>0.5), the clustering is robust.
        - If specific clusters show poor silhouette values, consider adjusting parameters
          or removing those clusters.
        - If the average silhouette is low (<0.3), consider different clustering parameters
          or preprocessing steps.
    """
    try:
        # -------------------------------
        # Generate color mapping
        # -------------------------------
        color_map = _color_map(clustered_df)

        # -------------------------------
        # Generate and return the silhouette plot figure
        # -------------------------------
        return _silhouette_fig(clustered_df, color_map)
    except Exception as e:
        logging.exception("Error generating silhouette plot:")
        raise e


def bars_fig(clusters_df: pd.DataFrame) -> go.Figure:
    """
    Generate a horizontal bar chart visualizing cluster sizes with topic labels.

    This function creates an informative bar chart that shows the relative sizes of each cluster,
    identified by both cluster ID and topic label (if available). It provides a quick overview
    of your clustering results, highlighting dominant topics and the distribution of data points
    across clusters.

    The visualization:
    - Displays clusters in order of size (smallest to largest by default)
    - Color-codes bars to match other visualizations (silhouette_fig and scatter_fig)
    - Shows exact counts as text annotations
    - Includes both cluster IDs and topic labels for clear identification

    Parameters:
        clusters_df (pd.DataFrame):
            DataFrame containing clustering results from run_clustering(). Must include a
            'cluster_id' column (int) with cluster identifiers (-1 for noise). If a 'topic'
            column (str) exists (from add_labels), these topics will be displayed alongside
            cluster IDs in the visualization.

    Returns:
        go.Figure:
            A Plotly Figure object containing a horizontal bar chart visualization.
            The chart provides a clear overview of cluster sizes and topics, with:

            - Horizontal bars representing each cluster, sorted by size (smallest to largest)
            - Color-coded bars matching colors used in other visualizations from this package
            - Text labels showing both cluster IDs and topic names
            - Count values displayed at the end of each bar
            - Dark theme styling for better visibility
            - Automatic height scaling based on the number of clusters

            The figure can be:
            - Displayed directly with fig.show()
            - Saved as an HTML file with fig.write_html("cluster_sizes.html")
            - Converted to static images with fig.write_image("cluster_sizes.png")
            - Customized further using Plotly's update_layout() and other methods
            - Embedded in Jupyter notebooks, dashboards, or web applications

            This visualization complements the scatter_fig() and silhouette_fig() by providing:
            - A quick summary of the cluster distribution in your dataset
            - A clear view of the relative sizes of each topic
            - An easy way to identify dominant vs. niche topics
            - A legend that pairs cluster IDs with their semantic labels

            For presentations and reports, consider placing this chart alongside the other
            visualizations to give viewers a complete picture of your clustering results.

    Notes:
        - Noise points (cluster_id = -1) are excluded from the visualization to focus on
          meaningful clusters.
        - The chart height automatically scales based on the number of clusters for optimal
          readability.
        - The consistent color coding across visualizations makes it easier to correlate
          information between different views of your data.
        - If the 'topic' column is missing, the function will create default labels in the
          format "Cluster X".

    Applications:
        - Quickly assess the distribution of data across topics
        - Identify dominant vs. niche topics
        - Communicate clustering results to stakeholders
        - Track changes in topic distribution when comparing different clustering runs
        - Provide context for more detailed visualizations
    """
    try:
        # -------------------------------
        # Generate color mapping
        # -------------------------------
        color_map = _color_map(clusters_df)

        # -------------------------------
        # Generate and return the bar chart figure
        # -------------------------------
        return _cluster_bar_chart_fig(clusters_df, color_map)
    except Exception as e:
        logging.exception("Error generating bar chart:")
        raise e


def validate_scatter_params(
    clusters_df: pd.DataFrame, content_col_name: str, wrap_width: int, id_col_name
) -> dict:
    """
    Validate parameters for the scatter_fig function.

    Args:
        clusters_df (pd.DataFrame): DataFrame containing clustering results.
        content_col_name (str): Name of the column with textual content.
        wrap_width (int): Maximum width for wrapping text in hover labels.
        id_col_name: Optional column name for record IDs.

    Raises:
        ValueError: If content_col_name is not found in clusters_df.
        ValueError: If id_col_name is provided but not found in clusters_df columns.

    Returns:
        dict: A dictionary with the key 'wrap_width' mapping to the validated wrap_width value (ensuring it is at least 20).
    """
    if content_col_name not in clusters_df.columns:
        raise ValueError(
            f"content_col_name '{content_col_name}' not found in clusters_df columns."
        )

    if id_col_name is not None and id_col_name not in clusters_df.columns:
        raise ValueError(
            f"id_col_name '{id_col_name}' not found in clusters_df columns."
        )

    if wrap_width < 20:
        logging.warning("wrap_width too small, defaulting to wrap_width=20")
        wrap_width = 20

    return {"wrap_width": wrap_width}


# ----------------------
# scatter_fig Functional Interface
# ----------------------


def scatter_fig(
    clusters_df: pd.DataFrame,
    content_col_name: str = "content",
    wrap_width: int = 100,
    id_col_name=None,
) -> go.Figure:
    """
    Generate an interactive scatter plot visualization of clustering results.

    This function creates a visually rich 2D or 3D plot (depending on the dimensionality of
    the reduced vectors) that shows the spatial distribution of clusters. It's a powerful
    tool for exploring clustering results, inspecting individual data points, and understanding
    the relationships between clusters.

    The plot includes:
    - Color-coded points for each cluster
    - Cluster labels positioned at cluster centroids
    - Interactive hover information showing content and metadata for each point
    - Noise points (if any) shown in a distinct color with reduced opacity

    Parameters:
        clusters_df (pd.DataFrame):
            DataFrame containing clustering results from run_clustering(). Must include columns:
            'cluster_id' (int): cluster identifiers with -1 for noise
            'reduced_vector' (list-like): reduced dimensional coordinates, must be 2D or 3D
                                          (e.g., [x, y] or [x, y, z])
            Also requires a content column (specified by content_col_name) containing text (str).
            If 'topic' column (str) exists, it will be used for cluster labels.

        content_col_name (str, default="content"):
            Name of the column containing text content that will be displayed when hovering
            over points in the scatter plot.

        wrap_width (int, default=100):
            Maximum width (in characters) for wrapping text in hover labels.
            Values below 20 will be automatically increased to 20.

        id_col_name (str, default=None):
            Optional column name for record IDs to include in hover information. For example,
            if your dataset contains movie descriptions, you might use the movie title column
            (e.g., id_col_name="title") to help identify each point in the visualization.
            This can be useful for tracking specific points of interest or connecting
            visualization data back to your original dataset.

    Returns:
        go.Figure:
            A Plotly Figure object containing an interactive scatter plot visualization.
            The function automatically creates either a 2D or 3D plot based on the
            dimensionality of the reduced vectors (2D for 2-dimensional vectors, 3D for
            3-dimensional vectors). For vectors with more than 3 dimensions, a warning
            message is displayed instead.

            The figure includes:
            - Color-coded points for each cluster, matching colors used in other visualizations
              from this package
            - Large text labels at cluster centroids showing cluster topics
            - Noise points (if any) displayed with reduced opacity
            - Interactive hover information displaying:
                * Cluster ID and topic (if available)
                * ID value (if id_col_name was provided)
                * Text content from the specified content column
            - A legend for identifying clusters
            - Dark theme styling for better visibility of cluster patterns

            The figure can be:
            - Displayed directly with fig.show()
            - Saved as an HTML file with fig.write_html("clusters.html")
            - Converted to static images with fig.write_image("clusters.png")
            - Customized further using Plotly's update_layout() and other methods
            - Embedded in Jupyter notebooks, dashboards, or web applications

            For 3D plots, additional interaction is available:
            - Rotation (click and drag)
            - Zoom (scroll)
            - Pan (right-click and drag)

            Note that for very large datasets (10,000+ points), the interactive performance
            may decrease. Consider sampling your data or using a static image export if needed.

    Notes:
        - The function automatically determines whether to create a 2D or 3D visualization
          based on the dimensionality of the 'reduced_vector' column.
        - For vectors with more than 3 dimensions, a warning is displayed (visualization
          is limited to 3D).
        - Noise points (cluster_id = -1) are shown with reduced opacity.
        - Hover information includes cluster ID, topic (if available), content, and optional ID.
        - The plot uses a dark theme optimized for visualizing clusters.
        - For large datasets, rendering may be slow. Consider sampling your data first
          if performance is an issue.

    Visualization tips:
        - Look for well-separated clusters, which indicate distinct topics
        - Clusters that overlap might benefit from parameter tuning
        - Examine outliers and noise points to identify potential improvements
        - Use the interactive features (zoom, rotation in 3D) to explore the structure
        - Hover over points to inspect individual content and verify cluster coherence
    """
    try:
        # Validate parameters first. Clunky code, but this also makes sure that wrap_width is at least 20.
        wrap_width = validate_scatter_params(
            clusters_df, content_col_name, wrap_width, id_col_name
        )["wrap_width"]

        # -------------------------------
        # Generate color mapping
        # -------------------------------
        color_map = _color_map(clusters_df)

        # -------------------------------
        # Use the ScatterPlot class to generate and return the scatter plot figure
        # -------------------------------
        scatter_plot = ScatterPlot(
            clusters_df=clusters_df,
            color_map=color_map,
            id_col_name=id_col_name,
            content_col_name=content_col_name,
            wrap_width=wrap_width,
        )
        return scatter_plot.create_figure()
    except Exception as e:
        logging.exception("Error generating scatter plot:")
        raise e
