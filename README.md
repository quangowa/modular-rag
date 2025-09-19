# Modular RAG with Haystack and Hypster

<p align="center">
  <img src="assets/modular-rag.png" alt="Modular RAG" width="100%">
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

## Installation and Usage

This project supports multiple package managers. Choose the method that best suits your environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/gilad-rubin/modular-rag.git
   cd modular-rag

2. Install dependencies using one of the following methods:

### uv

```bash
uv run --with jupyter jupyter lab
```

### Conda

```bash
conda env create -n modular-rag python=3.10 -y
conda activate modular-rag
pip install -r requirements.txt

jupyter lab
```

### pip

```bash
pip install -r requirements.txt
jupyter lab
```

3. Open the main notebook and follow the instructions to experiment with different RAG configurations.


### Running local Ollama models

Local LLMs served by [Ollama](https://ollama.com/) are supported alongside the existing OpenAI and Anthropic integrations.

1. Install the optional Haystack integration package:

   ```bash
   pip install ollama-haystack
   ```

2. Ensure the Ollama daemon is running on your machine (for example `ollama serve`) and that the desired models (such as `llama3`, `llama3.1`, or `mistral`) are available.
3. In the notebooks select the `ollama` provider when configuring an LLM. The configuration lets you override the base URL (default `http://localhost:11434`) or you can set the `OLLAMA_BASE_URL` environment variable before launching Jupyter.
4. The quick-start notebook (`1_hardcoded.ipynb`) exposes an `llm_backend` toggle so you can flip between OpenAI and Ollama when you want to exercise the local path.

## Code Examples

Here's a basic example of how to use the modular RAG system:

```python
results = modular_rag(
    values={
        "indexing.enrich_doc_w_llm": True,
        "indexing.llm.model": "gpt-4o-mini",
        "document_store_type": "qdrant",
        "retrieval.bm25_weight": 0.8,
        "embedder_type": "fastembed",
        "reranker.model": "tiny-bert-v2",
        "response.llm.model": "haiku",
        "indexing.splitter.split_length": 6,
        "reranker.top_k": 3
    },
)
indexing_pipeline = results["indexing_pipeline"]
indexing_pipeline.warm_up()
file_paths = ["data/raw/modular_rag.pdf"]
for file_path in file_paths:  # this can be parallelized
    indexing_pipeline.run({"loader": {"sources": [file_path]}})
query = "What are the 6 main modules of the modular RAG framework?"

pipeline = results["pipeline"]
pipeline.warm_up()
response = pipeline.run({"query": {"text": query}}, include_outputs_from=["prompt_builder", "docs_for_generation"])
print(response["llm"]["replies"][0])
