from pathlib import Path

from src.retrieval.bm25_retriever import BM25MedicalRetriever, load_articles_from_json


def main() -> None:
    data_path = Path("data/raw/pubmed_articles.json")
    articles = load_articles_from_json(data_path)

    retriever = BM25MedicalRetriever(
        articles=articles,
        k1=1.2,
        b=0.5,
    )

    queries = [
        "Çölyak hastalığı tanı kriterleri nelerdir?",
        "What are the latest guidelines for managing type 2 diabetes?",
    ]

    for query in queries:
        print("=" * 100)
        print(f"QUERY: {query}")
        print("-" * 100)

        explanation = retriever.explain_query(query)
        print(f"Normalized query: {explanation['normalized_query']}")
        print(f"Tokens: {explanation['tokens']}")
        print(f"k1={explanation['k1']} | b={explanation['b']}")
        print()

        results = retriever.search(query, top_k=5)

        for r in results:
            print(f"[{r.rank}] score={r.score:.4f} | PMID={r.pmid} | Year={r.year}")
            print(f"Title: {r.title}")
            #print(f"Journal: {r.journal}")
            #print(f"DOI: {r.doi}")
            print(f"Matched terms: {r.matched_terms}")
            print(f"Preview: {r.retrieval_text_preview}")
            print("-" * 100)


if __name__ == "__main__":
    main()