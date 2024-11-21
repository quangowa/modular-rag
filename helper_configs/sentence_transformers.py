from hypster import HP


def sentence_transformers_config(hp: HP):
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    model_types = ["sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/paraphrase-MiniLM-L6-v2"]
    emb_model = hp.select(model_types, default="sentence-transformers/paraphrase-MiniLM-L6-v2")

    document_store = InMemoryDocumentStore()
    doc_embedder = SentenceTransformersDocumentEmbedder(model=emb_model, batch_size=hp.int(16, min=2, max=256))
    text_embedder = SentenceTransformersTextEmbedder(model=emb_model)
    retriever = InMemoryEmbeddingRetriever(document_store, top_k=hp.int(10))

    embedding_str = "embedding"
