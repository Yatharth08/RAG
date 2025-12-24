from ingestion.loader import DocumentLoader
from chunking.chunking import SemanticChunker
from embedding.embedding import Embedding
from vectorstore.vector_db import VectorDB
from sentence_transformers import SentenceTransformer
from retrieval.vector.faiss_retriever import SearchFaiss
from retrieval.lexical.bm25_retriever import BM25Retriever
from retrieval.fusion.rrf import RRF
from models.llm import LLM
from models.evaluate import LLMAsAJudge
from testset.testset import test_set
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

    llm = LLM()
    faiss_retriever = SearchFaiss(embedder.embedder)
    bm25_retriever = BM25Retriever(chunks)
    rrf = RRF()
    llm_as_judge = LLMAsAJudge()

    faithfulness_scores = []
    answer_relevance_scores = []
    for item in test_set:
        question = item["question"]
        ground_truth = item["answer"]
        results = faiss_retriever.search_faiss_context(question, index)
        bm25_results = bm25_retriever.lexical_search(question)
        context = rrf.rrf_fusion_context(results, bm25_results, chunks)
        model_answer = llm.generate_answer_with_llm(question, context)
        judge = llm_as_judge.llm_judge(question, ground_truth, model_answer, context)
        faithfulness_scores.append(judge["faithfulness"])
        answer_relevance_scores.append(judge["answer_relevance"])
    final_metrics = {
        "faithfulness_avg": sum(faithfulness_scores) / len(faithfulness_scores),
        "answer_relevance_avg": sum(answer_relevance_scores) / len(answer_relevance_scores)
    }
    print(final_metrics)
if __name__ == "__main__":
    main()
    