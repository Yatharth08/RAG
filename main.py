from ingestion.loader import DocumentLoader
from chunking.chunking import SemanticChunker
from embedding.embedding import Embedding
from vectorstore.vector_db import VectorDB
from sentence_transformers import SentenceTransformer
def main():
    loader = DocumentLoader()
    text = loader.load_data("assets/TRI Q3 2025 Earnings Presentation_Final.pdf")

    chunker = SemanticChunker()
    chunks = chunker.chunk(text)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = Embedding(embedder)
    chunk_embeddings = embedder.embed(chunks)

    vectordb = VectorDB()
    index = vectordb.store_embeddings_in_vectodb(chunk_embeddings)
if __name__ == "__main__":
    main()
    