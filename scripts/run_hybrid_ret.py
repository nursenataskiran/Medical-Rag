from pathlib import Path
from typing import Dict

from src.retrieval.bm25_retriever import BM25MedicalRetriever, load_articles_from_json
from src.retrieval.semantic_retriever import SemanticMedicalRetriever
from src.retrieval.hybrid_retriever import HybridMedicalRetriever


DATA_PATH = Path("data/raw/pubmed_articles.json")
RETRIEVAL_DEPTH = 10
DISPLAY_TOP_K = 5


def build_lookup(results) -> Dict[str, dict]:
    lookup = {}
    for r in results:
        if r.pmid is None:
            continue
        lookup[str(r.pmid)] = {
            "rank": r.rank,
            "score": r.score,
            "title": r.title,
        }
    return lookup


def main() -> None:
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

    queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çölyak hastalığı tanı kriterleri nelerdir?",
    ]

    for query in queries:
        print("=" * 120)
        print(f"QUERY: {query}")
        print("-" * 120)

        bm25_explanation = bm25_retriever.explain_query(query)
        semantic_explanation = semantic_retriever.explain_query(query)

        print(f"BM25 normalized query: {bm25_explanation['normalized_query']}")
        print(f"BM25 tokens: {bm25_explanation['tokens']}")
        print(f"BM25 params: k1={bm25_explanation['k1']} | b={bm25_explanation['b']}")
        print(f"Semantic normalized query: {semantic_explanation['normalized_query']}")
        print(f"Semantic model: {semantic_explanation['model_name']}")
        print()

        bm25_results = bm25_retriever.search(query, top_k=RETRIEVAL_DEPTH)
        semantic_results = semantic_retriever.search(query, top_k=RETRIEVAL_DEPTH)
        hybrid_results = hybrid_retriever.search(query, top_k=DISPLAY_TOP_K)

        bm25_lookup = build_lookup(bm25_results)
        semantic_lookup = build_lookup(semantic_results)

        for item in hybrid_results:
            pmid_key = str(item.pmid) if item.pmid is not None else None

            bm25_info = bm25_lookup.get(pmid_key)
            semantic_info = semantic_lookup.get(pmid_key)

            bm25_rank = bm25_info["rank"] if bm25_info else "-"
            bm25_score = f"{bm25_info['score']:.4f}" if bm25_info else "-"
            semantic_rank = semantic_info["rank"] if semantic_info else "-"
            semantic_score = f"{semantic_info['score']:.4f}" if semantic_info else "-"

            print(
                f"[{item.rank}] rrf_score={item.score:.6f} | PMID={item.pmid} | Year={item.year}"
            )
            print(f"Title: {item.title}")
            print(f"Journal: {item.journal}")
            print(f"Matched terms: {item.matched_terms}")
            print(
                f"Source signals: BM25(rank={bm25_rank}, score={bm25_score}) | "
                f"Semantic(rank={semantic_rank}, score={semantic_score})"
            )
            print(f"Preview: {item.retrieval_text_preview}")
            print("-" * 120)

        print()


if __name__ == "__main__":
    main()