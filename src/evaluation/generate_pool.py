from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from src.retrieval.bm25_retriever import BM25MedicalRetriever, load_articles_from_json
from src.retrieval.semantic_retriever import SemanticMedicalRetriever
from src.retrieval.hybrid_retriever import HybridMedicalRetriever


DATA_PATH = Path("data/raw/pubmed_articles.json")
OUTPUT_PATH = Path("data/eval/evaluation_pool_blind.csv")
RETRIEVAL_DEPTH = 10

QUERIES = [
    "What are the latest guidelines for managing type 2 diabetes?",
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
    "Iron supplementation dosing for anemia during pregnancy",
    "Çölyak hastalığı tanı kriterleri nelerdir?",
    "Antibiotic resistance patterns in community acquired pneumonia",
]


def result_to_doc(result: Any) -> Dict[str, str]:
    return {
        "pmid": str(result.pmid) if result.pmid is not None else "",
        "title": (result.title or "").strip(),
        "abstract": (result.retrieval_text_preview or "").strip(),
    }


def merge_unique_docs_for_query(*result_lists: List[Any]) -> List[Dict[str, str]]:
    """
    Deduplicate only within a single query pool.
    The same PMID may appear again under a different query.
    """
    pooled: Dict[str, Dict[str, str]] = {}
    fallback_counter = 0

    for results in result_lists:
        for r in results:
            doc = result_to_doc(r)

            pmid = doc["pmid"]
            if pmid:
                doc_id = f"pmid:{pmid}"
            else:
                fallback_counter += 1
                doc_id = f"no_pmid:{fallback_counter}:{doc['title'][:80]}"

            if doc_id not in pooled:
                pooled[doc_id] = doc

    return list(pooled.values())


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    articles = load_articles_from_json(DATA_PATH)

    bm25_retriever = BM25MedicalRetriever(
        articles=articles,
        k1=1.5,
        b=0.5,
    )
    semantic_retriever = SemanticMedicalRetriever(articles)
    hybrid_retriever = HybridMedicalRetriever(
        bm25_retriever=bm25_retriever,
        semantic_retriever=semantic_retriever,
        k=20,
    )

    all_rows: List[Dict[str, str]] = []

    for query in QUERIES:
        print(f"Running query: {query}")

        bm25_results = bm25_retriever.search(query, top_k=RETRIEVAL_DEPTH)
        semantic_results = semantic_retriever.search(query, top_k=RETRIEVAL_DEPTH)
        hybrid_results = hybrid_retriever.search(query, top_k=RETRIEVAL_DEPTH)

        pooled_docs = merge_unique_docs_for_query(
            bm25_results,
            semantic_results,
            hybrid_results,
        )

        print(f"  -> pooled {len(pooled_docs)} unique docs for this query")

        for doc in pooled_docs:
            all_rows.append(
                {
                    "query": query,
                    "pmid": doc["pmid"],
                    "title": doc["title"],
                    "abstract": doc["abstract"],
                    "relevance": "",
                }
            )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "pmid", "title", "abstract", "relevance"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f"Saved blind evaluation pool to: {OUTPUT_PATH}")
    print(f"Total pooled rows: {len(all_rows)}")


if __name__ == "__main__":
    main()