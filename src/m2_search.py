"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Implement Vietnamese word segmentation
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")

from rank_bm25 import BM25Okapi

class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []

        for chunk in chunks:
            text = chunk["text"]
            # Vietnamese segmentation → space-separated tokens
            tokenized = segment_vietnamese(text).split()
            self.corpus_tokens.append(tokenized)

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            raise ValueError("BM25 index has not been built. Call index() first.")

        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.documents[idx]
            score = float(scores[idx])

            results.append(
                SearchResult(
                    text=chunk["text"],
                    score=score,
                    metadata=chunk.get("metadata", {}),
                    method="bm25",
                )
            )

        return results


from qdrant_client.models import Distance, VectorParams, PointStruct

class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        # 1. Create or recreate collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

        if not chunks:
            return

        # 2. Encode texts
        texts = [c["text"] for c in chunks]
        embeddings = self._get_encoder().encode(
            texts, show_progress_bar=True
        )

        # 3. Build points
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            payload = {
                **chunk.get("metadata", {}),
                "text": chunk["text"],
            }

            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        # 4. Upsert into Qdrant
        self.client.upsert(
            collection_name=collection,
            points=points,
        )

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()

        repsponse = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )

        hits = repsponse.points

        results = []
        for hit in hits:
            payload = hit.payload
            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=float(hit.score),
                    metadata=payload,
                    method="dense",
                )
            )


        return results



def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """
    Merge ranked lists using Reciprocal Rank Fusion (RRF):
        score(d) = sum_i 1 / (k + rank_i)

    Args:
        results_list: List of ranked result lists (BM25, Dense, ...)
        k: RRF constant (larger k → smoother fusion)
        top_k: Number of final results to return

    Returns:
        List of SearchResult with method="hybrid"
    """
    rrf_scores: dict[str, dict] = {}

    # -------- 1. Accumulate RRF scores --------
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text  # dùng text làm key (đơn giản & hiệu quả)

            if key not in rrf_scores:
                rrf_scores[key] = {
                    "score": 0.0,
                    "result": result,
                }

            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    # -------- 2. Sort by RRF score --------
    fused = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    # -------- 3. Build final SearchResult list --------
    final_results = []
    for item in fused[:top_k]:
        r = item["result"]
        final_results.append(
            SearchResult(
                text=r.text,
                score=item["score"],
                metadata=r.metadata,
                method="hybrid",
            )
        )

    return final_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
