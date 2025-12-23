from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self,model_name: str = "all-MiniLM-L6-v2",similarity_threshold: float = 0.75, max_sentences_per_chunk: int = 10,):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def chunk(self, text):
        if not text or not text.strip():
            return []
        
        # Function from NLTK that splits a paragraph of text into individual sentences.
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        
        # for each sentence create embeddings
        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        chunks = []
        current_chunk = [sentences[0]]
        for i in range(1, len(sentences)):
            # cosine_similarity return 2d matrix eg. [[0.87]]
            # extract 0 row and 0 column
            sim = cosine_similarity([embeddings[i-1]],[embeddings[i]])[0][0]
            if sim >= self.similarity_threshold and len(current_chunk) < self.max_sentences_per_chunk:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
        # Last chunk was not attended inside loop
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks