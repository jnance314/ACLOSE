# ACLOSE
Automatic Clustering and Labeling Of Semantic Embeddings


## What it does
This package is a tool for quick EDA of emergent topics among your semantic embeddings.
### Problem
- I have all these embedding vectors of text documents. What are the general topics present in the text corpus?
### Solution
- ACLOSE will cluster your embeddings and then label the clusters using an LLM.
- Instead of throwing a random sample of embeddings from each cluster at an LLM, ACLOSE uses stratified sampling and refinement to ensure that the topic labels balance generalization and specificity.

## Algorithms
- PCA
- UMAP
- HDBSCAN

## C++ compiler required

ACLOSE uses UMAP, which requires a c++ compiler to run. Before installing ACLOSE, ensure you have:

- Windows: Microsoft Visual C++ Build Tools

- Linux: GCC/G++ compiler (`sudo apt-get install build-essential` on Ubuntu)

- macOS: Xcode Command Line Tools (`xcode-select --install`)

## Tip for building in Docker
Add this to your dockerfile:
```Dockerfile
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libx11-dev \
    libxrandr-dev \
    libxext-dev \
    libxi-dev \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
```

## Notebook demo
[Colab notebook with three example uses](https://colab.research.google.com/drive/1UsXnxj2aT2VmL7HP2QiAbvJJk_n-eIhr?usp=sharing)

## Quickstart
```python
pip install aclose
```

## Number of LLM calls
- 2 LLM calls per cluster
- User can choose the model


## Instructions for use
Assume df has columns: 
- content_str
- embedding_vector

Gets additional columns after applying .label(df, data_description)
- cluster_id
- topic_label
- membership_score
- outlier_score
- silhouette_score
- reduced_vector
