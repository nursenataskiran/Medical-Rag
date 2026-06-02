from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    rank: int
    score: float
    pmid: Optional[str]
    title: str
    journal: Optional[str]
    year: Optional[Any]
    doi: Optional[str]
    matched_terms: List[str]
    normalized_query: str
    retrieval_text_preview: str


class HybridMedicalRetriever:
    def __init__(self, bm25_retriever, semantic_retriever, k: int = 20):
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever
        self.k = k

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=top_k)
        semantic_results = self.semantic.search(query, top_k=top_k)

        rrf_scores = defaultdict(float)
        doc_store = {}

        for results in [bm25_results, semantic_results]:
            for rank, doc in enumerate(results, start=1):
                doc_id = str(doc.pmid)

                if doc_id not in doc_store:
                    doc_store[doc_id] = doc

                rrf_scores[doc_id] += 1.0 / (self.k + rank)

        ranked_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:top_k]

        fused_results = []

        for rank, doc_id in enumerate(ranked_ids, start=1):
            doc = doc_store[doc_id]

            fused_results.append(
                SearchResult(
                    rank=rank,
                    score=float(rrf_scores[doc_id]),
                    pmid=doc.pmid,
                    title=doc.title,
                    journal=doc.journal,
                    year=doc.year,
                    doi=doc.doi,
                    matched_terms=doc.matched_terms,
                    normalized_query=query,
                    retrieval_text_preview=doc.retrieval_text_preview,
                )
            )

        return fused_results