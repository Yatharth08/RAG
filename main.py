from ingestion.loader import DocumentLoader
from chunking.chunking import SemanticChunker
from embedding.embedding import Embedding
def main():
    loader = DocumentLoader()
    text = loader.load_data("assets/TRI Q3 2025 Earnings Presentation_Final.pdf")

    chunker = SemanticChunker()
    chunks = chunker.chunk(text)

    embedder = Embedding()
    chunk_embeddings = embedder.embed(chunks)

if __name__ == "__main__":
    main()
    