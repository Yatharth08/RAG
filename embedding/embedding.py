from sentence_transformers import SentenceTransformer
class Embedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self,chunks):
        chunk_embeddings = self.model.encode(chunks, normalize_embeddings=True)
        return chunk_embeddings