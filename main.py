from ingestion.loader import DocumentLoader
from chunking.chunking import SemanticChunker
def main():
    loader = DocumentLoader()
    text = loader.load_data("assets/TRI Q3 2025 Earnings Presentation_Final.pdf")

    chunker = SemanticChunker()
    chunks = chunker.chunk(text)
    
if __name__ == "__main__":
    main()
    