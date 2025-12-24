class RRF:
    def rrf_fusion_context(self, faiss_results, bm25_results, chunks, k=60, top_k=5):
        scores = {}
        # FAISS contributions
        # faiss_results = [
        #   {"index": 2, "rank": 1},
        #   {"index": 5, "rank": 2},
        #   {"index": 3, "rank": 3}
        # ]
        for r in faiss_results:
            idx = r["index"]
            rank = r["rank"]
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
        # scores = {
        #   2: 1/(60+1) = 0.0164,
        #   5: 1/(60+2) = 0.0161,
        #   3: 1/(60+3) = 0.0159
        # }

        # BM25 contributions
        for r in bm25_results:
            idx = r["index"]
            rank = r["rank"]
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
        # scores = {
        #   2: 0.0164 + 1/(60+2) = 0.0325,
        #   5: 0.0161,
        #   3: 0.0159 + 1/(60+1) = 0.0323,
        #   7: 1/(60+3) = 0.0159
        # }

        # Sort by fused score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Extract chunk indices eg: [2, 3, 5, 7]
        top_indices = [idx for idx, _ in fused]

        # Build context string
        context = "\n\n".join(chunks[idx] for idx in top_indices)

        return context