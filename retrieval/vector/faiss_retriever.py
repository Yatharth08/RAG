from sentence_transformers import SentenceTransformer
class SearchFaiss:
    def __init__(self, embedder):
        self.embedder = embedder

    def search_faiss_context(self, query, index, top_k=5):
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        # Score with chunks index in descending order
        scores, indices = index.search(q_emb, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            results.append({
                "index": int(idx),
                "rank": rank + 1
            })
        return results