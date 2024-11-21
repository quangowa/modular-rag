from hypster import HP


def fastembed_config(hp: HP):
    from fastembed import SparseTextEmbedding
    from haystack_integrations.components.embedders.fastembed import (
        FastembedSparseDocumentEmbedder,
        FastembedSparseTextEmbedder,
    )
    from haystack_integrations.components.retrievers.qdrant import QdrantSparseEmbeddingRetriever
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    models = [dct["model"] for dct in SparseTextEmbedding.list_supported_models()]
    document_store = QdrantDocumentStore(":memory:", recreate_index=True, use_sparse_embeddings=True)

    emb_model = hp.select(models)
    doc_embedder = FastembedSparseDocumentEmbedder(model=emb_model, batch_size=hp.int(16, min=2, max=256))
    retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
    text_embedder = FastembedSparseTextEmbedder(model=emb_model)

    embedding_str = "sparse_embedding"
