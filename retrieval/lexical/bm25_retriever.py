from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
class BM25Retriever:
    def __init__(self, chunks):
        tokenized = [word_tokenize(c.lower()) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def lexical_search(self, query, top_k=5):
        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)

        # Pair document index with score
        # eg: [(0, 0.0), (1, 4.21), (2, 1.98), (3, 6.43), (4, 0.0)]
        ranked = sorted(list(enumerate(scores)),key=lambda x: x[1],reverse=True)[:top_k]

        results = []
        for i, (idx, score) in enumerate(ranked, start=1):
            results.append({
                "index": idx,
                "rank": i
            })
        return results