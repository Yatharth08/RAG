from sentence_transformers import SentenceTransformer
class Embedding:
    def __init__(self, embedder):
        self.embedder = embedder

    def embed(self,chunks):
        chunk_embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
        return chunk_embeddings