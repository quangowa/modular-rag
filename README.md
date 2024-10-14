# Modular RAG with Haystack and Hypster

<p align="center">
  <img src="assets/modular-rag.png" alt="Modular RAG" width="70%">
</p>

This project implements the concepts described in the paper: [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/abs/2407.21059) by Yunfan Gao et al. 

For a detailed walkthrough, refer to this [Medium article](https://medium.com/p/d2f0ecc88b8f).

## Key Objectives

- Decompose RAG (Retrieval-Augmented Generation) into its fundamental components using **Haystack**.
- Utilize **Hypster** to manage a "hyper-space" of potential RAG configurations.
- Facilitate easy swapping and experimentation with various implementations.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Code Examples](#code-examples)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

This project supports multiple package managers. Choose the method that best suits your environment:

### Poetry

```bash
poetry install
```

### pip

```bash
pip install -r requirements.txt
```

### Conda

```bash
conda env create -n modular-rag python=3.10 -y
conda activate modular-rag
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/gilad-rubin/modular-rag.git
   cd modular-rag
   ```

2. Install dependencies using your preferred method from the [Installation](#installation) section.

3. Launch Jupyter Lab:
   ```bash
   jupyter lab
   ```

4. Open the main notebook and follow the instructions to experiment with different RAG configurations.

## Code Examples

Here's a basic example of how to use the modular RAG system:

```python
results = rag_config(
    selections={
        "indexing.enrich_doc_w_llm": True,
        "indexing.llm.model": "gpt-4o-mini",
        "document_store_type": "qdrant",
        "retrieval.bm25_weight": 0.8,
        "embedder_type": "fastembed",
        "reranker.model": "tiny-bert-v2",
        "response.llm.model": "haiku",
    },
    overrides={"indexing.splitter.split_length": 6, "reranker.top_k": 3},
)
indexing_pipeline = results["indexing_pipeline"]
indexing_pipeline.warm_up()

file_paths = ["data/raw/modular_rag.pdf", "data/raw/enhancing_rag.pdf"]
for file_path in file_paths:  # this can be parallelized
    indexing_pipeline.run({"loader": {"sources": [file_path]}})

query = "What are the 6 main modules of the modular RAG framework?"

pipeline = results["pipeline"]
pipeline.warm_up()
response = pipeline.run({"query": {"text": query}}, include_outputs_from=["prompt_builder", "docs_for_generation"])
response["llm"]