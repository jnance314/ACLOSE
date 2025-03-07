# ACLOSE 🔍✨ Automatic Clustering and Labeling Of Semantic Embeddings

<div align="center">
  
[![PyPI version](https://img.shields.io/badge/PyPi-aclose-green)](https://pypi.org/project/aclose/)
[![Version](https://img.shields.io/badge/Version-0.1.0-blue)](https://pypi.org/project/aclose/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ACLOSE-blue?style=flat&logo=github)](https://github.com/jnance314/aclose)

</div>

## 🌟 What is ACLOSE?

ACLOSE is a powerful machine learning library that automates the discovery, labeling, and visualization of topics within text data. It combines cutting-edge dimensionality reduction, clustering, and large language models to transform raw embeddings into meaningful, labeled topics with minimal code.

Think of it as **automatic topic discovery without the headaches**.

![Visualize Clusters](https://storage.googleapis.com/portfolio-site-assets/aclose_assets/demo.gif)

## 🔥 Why Use ACLOSE?

### The Problem ACLOSE Solves

- 📊 **Embedding vectors by themselves aren't helpful** for understanding content themes
- 🧩 **Manual topic discovery is tedious** and doesn't scale to large datasets
- 🏷️ **Labeling clusters is subjective** and time-consuming
- ⚙️ **Tuning clustering algorithms is complex** and requires expertise

### ACLOSE's Solution

ACLOSE offers a streamlined, three-step process:

1. **Cluster** text embeddings using optimized hyperparameters
2. **Label** the clusters with semantic topics using LLMs
3. **Visualize** the results with publication-quality interactive plots

No more guessing at parameters or manually interpreting cluster contents!

## ✨ Key Features

- **🤖 End-to-End Automation**: From raw embeddings to labeled topics in just a few lines of code
- **📐 Multi-Objective Optimization**: Intelligent hyperparameter tuning with Pareto front selection
- **🎯 Smart LLM-Based Labeling**: Two-pass approach with core and peripheral point sampling for accurate topics
- **📊 Interactive Visualizations**: Ready-to-use cluster exploration with minimal setup
- **⚡ Production Ready**: Trained models that can be reused for classifying new data
- **📈 Drift Monitoring**: Tools to detect when clustering models need retraining

## 📦 Installation

### Prerequisites

Before installing, make sure you have a C++ compiler:

- **Windows**: Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **Linux**: `sudo apt-get install build-essential`
- **macOS**: Install Xcode Command Line Tools with `xcode-select --install`

### Install from PyPI

```bash
pip install aclose
```

## 🚀 Quick Start

### 1. Cluster your embeddings

```python
import pandas as pd
from aclose import run_clustering

# Example DataFrame with embeddings
df = pd.DataFrame({
    "content": ["Text document 1", "Text document 2", "Text document 3"],
    "embedding_vector": [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
})

# Run clustering with optimized parameters
result = run_clustering(df)

# Get the clustered dataframe
clustered_df = result["clustered_df"]
```

### 2. Label your clusters

```python
from aclose import add_labels

# Generate semantic topic labels for clusters
label_result = add_labels(
    cluster_df=clustered_df,
    data_description="Dataset of scientific paper abstracts",
    llm_model="o1-mini"  # Use OpenAI models
)

# Get labeled dataframe and mapping
labeled_df = label_result["dataframe"]
topic_mapping = label_result["labels_dict"]

print(topic_mapping)  # {0: "Machine Learning Applications", 1: "Climate Change Research", ...}
```

### 3. Visualize your topics

```python
from aclose import silhouette_fig, scatter_fig, bars_fig

# Generate and display three complementary visualizations
silhouette_fig(labeled_df).show()  # Assess cluster quality
scatter_fig(labeled_df, content_col_name="content").show()  # Explore semantic space
bars_fig(labeled_df).show()  # View topic distribution
```

## 📊 Visualizations

ACLOSE provides three powerful visualizations to help you understand your data:

### 🔍 Cluster Exploration (3D/2D Interactive)

Explore the semantic relationships between your documents in an interactive 3D or 2D visualization. Each point represents a document, color-coded by cluster, with topics labeled at cluster centers.

![Visualize Clusters](https://storage.googleapis.com/portfolio-site-assets/aclose_assets/scatter.png)

### 📊 Topic Distribution

See the relative sizes of each topic in your dataset with a clear, color-coded bar chart. Quickly identify dominant themes and niche topics.

![Topic Prevalence](https://storage.googleapis.com/portfolio-site-assets/aclose_assets/bars.png)

### 📈 Cluster Quality Assessment

Evaluate the quality of your clustering with a silhouette plot. Higher values indicate better-defined clusters, helping you assess the reliability of your topics.

![Cluster Quality](https://storage.googleapis.com/portfolio-site-assets/aclose_assets/silhouette.png)

## 🧠 Use Cases

### 1. Quick Exploratory Data Analysis

Instantly discover the main themes in your text corpus without manual annotation or parameter tuning.

```python
from aclose import run_clustering, add_labels, scatter_fig

result = run_clustering(df)
labeled = add_labels(result["clustered_df"])
scatter_fig(labeled["dataframe"]).show()
```

### 2. Experimentation and Refinement

Try different dimensionality settings before committing to expensive labeling operations:

```python
# Try 2D clustering (good for visualization)
clustering_2d = run_clustering(df, dims=2)

# Try 3D clustering (better balance of viz & quality)
clustering_3d = run_clustering(df, dims=3)

# Let the algorithm find optimal dimensions
clustering_nd = run_clustering(df, dims=None)

# Compare metrics
print(clustering_2d["metrics_dict"])
print(clustering_3d["metrics_dict"])
print(clustering_nd["metrics_dict"])

# Choose the best and label it
best_clustering = clustering_3d  # based on metrics
labeled = add_labels(best_clustering["clustered_df"])
```

### 3. Production ML Pipeline Integration

Reuse trained models to classify new data and monitor distribution drift:

```python
# Train initial models
clustering = run_clustering(training_df)
labeled = add_labels(clustering["clustered_df"])

# Extract models for reuse
umap_model = clustering["umap_model"]
hdbscan_model = clustering["hdbscan_model"]
topic_mapping = labeled["labels_dict"]

# Apply to new data
new_embeddings = get_embeddings(new_df)
reduced_vectors = umap_model.transform(new_embeddings)
new_labels, probabilities = hdbscan.approximate_predict(hdbscan_model, reduced_vectors)
new_df["topic"] = [topic_mapping.get(label, "Unknown") for label in new_labels]
```

## ⚙️ How It Works: The Magic Behind ACLOSE

ACLOSE isn't just a simple pipeline—it employs sophisticated techniques to produce high-quality topic clusters:

### 1. Smart Dimensionality Reduction

- **PCA Preprocessing**: Optional noise reduction that preserves a target explained variance ratio
- **UMAP Transformation**: Non-linear dimensionality reduction that maintains local structure

### 2. Intelligent Clustering

- **HDBSCAN**: Density-based clustering that automatically finds natural groupings
- **Branch Detection**: Optional hierarchical structure identification to find sub-topics

### 3. Advanced Hyperparameter Optimization

- **Triple-Objective Pareto Front**: Balances silhouette score, noise ratio, and cluster count
- **TOPSIS Selection**: Chooses the optimal configuration from the Pareto front

### 4. Two-Pass Topic Labeling

- **Core Point Sampling**: Identifies representative documents from each cluster's center
- **Stratified Peripheral Sampling**: Refines topics based on the full distribution of documents
- **Intelligent Prompting**: Guides the LLM to generate specific, distinctive topic labels

## 📖 Quick Documentation
For detailed documentation, including guidance on all hyperparameters, see [DOCUMENTATION.md](DOCUMENTATION.md).
Alternatively, if you're in a hurry, you can chat with the code using a [(gimmicky) custom GPT](https://chatgpt.com/g/g-67ca677711e08191b799250928221fde-aclose-documentation-bot).

### Core Functions

#### `run_clustering`

Performs optimized clustering on embeddings and returns models and results.

```python
result = run_clustering(
    filtered_df,                     # DataFrame with embedding_vector column
    min_clusters=3,                  # Minimum acceptable clusters
    max_clusters=25,                 # Maximum acceptable clusters
    dims=3,                          # Target dimensionality (None to optimize)
    target_pca_evr=0.9,              # PCA explained variance ratio target
    hdbscan_outlier_threshold=10,    # Percentile for core point detection
    # Many more configurable parameters...
)
```

Returns a dictionary with:

- `clustered_df`: DataFrame with cluster assignments and metadata
- `umap_model`: Fitted UMAP model for reuse
- `hdbscan_model`: Fitted HDBSCAN model
- `pca_model`: Fitted PCA model (if used)
- `metrics_dict`: Clustering quality metrics
- `branch_detector`: Branch detector (if used)

#### `add_labels`

Generates semantic topic labels for clusters using LLMs.

```python
result = add_labels(
    cluster_df,                     # DataFrame from run_clustering
    llm_model="o1-mini",            # LLM to use for labeling
    language="english",             # Output language
    data_description="Scientific papers", # Context for the LLM
    content_col_name="abstract",    # Column with text content
    # More configuration options...
)
```

Returns a dictionary with:

- `dataframe`: Original DataFrame with added 'topic' column
- `labels_dict`: Mapping from cluster_id to topic label

#### Visualization Functions

- `silhouette_fig(df)`: Creates a silhouette plot for evaluating cluster quality
- `scatter_fig(df)`: Creates a 2D/3D scatter plot of document clusters
- `bars_fig(df)`: Creates a bar chart of topic distribution

## 🔧 Requirements

- Python 3.10+
- OpenAI API key (for LLM-based labeling)
- Helicone API key (optional, for API call tracking)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## 🙏 Acknowledgments

- Developed and maintained by [Joe Nance](mailto:joe@nceno.app)
- Built on the shoulders of giants: UMAP, HDBSCAN, Optuna, and OpenAI

---

## 💡 Request for Features

Here is a list of features that we are planning to add in the future. If you would like to take up any of these features, please create an issue and assign it to yourself:

1. Support for non-openai and OSS LLMs via LiteLLM
2. More than two passes for topic label refinement
3. Support for other clustering algorithms
5. Lightweight (non-langchain) utilities for creating chunking and embedding
6. More options for chart visualization
