import faiss
class VectorDB:

    def store_embeddings_in_vectodb(self, chunk_embeddings):
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(chunk_embeddings)
        return index