"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, statistics
from dataclasses import dataclass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Load cross-encoder model
            # Option A: from FlagEmbedding import FlagReranker
            #           self._model = FlagReranker(self.model_name, use_fp16=True)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        # Implement reranking
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        
        # 3. scores = model.compute_score(pairs)  # FlagReranker
        scores = model.predict(pairs)      # CrossEncoder

        combined = []
        for score, doc in zip(scores, documents):
            combined.append({
                "rerank_score": float(score),
                "document": doc
            })

        # -------- 4. Sort by rerank score --------
        combined.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # -------- 5. Build final RerankResult --------
        results = []
        for rank, item in enumerate(combined[:top_k], start=1):
            doc = item["document"]
            results.append(
                RerankResult(
                    text=doc["text"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=item["rerank_score"],
                    metadata=doc.get("metadata", {}),
                    rank=rank,
                )
            )

        return results



class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # TODO (optional): from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    # Implement benchmark
    times = []
    for _ in range(n_runs):
      start = time.perf_counter()
      reranker.rerank(query, documents)
      times.append((time.perf_counter() - start) * 1000)  # ms
    return {"avg_ms": statistics.mean(times), "min_ms": min(times), "max_ms": max(times)}
    return {"avg_ms": 0, "min_ms": 0, "max_ms": 0}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
