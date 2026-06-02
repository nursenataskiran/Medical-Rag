from src.retrieval.semantic_retriever import SemanticMedicalRetriever, load_articles_from_json

JSON_PATH = "data/raw/pubmed_articles.json"  # bunu kendi dosya yoluna göre değiştir

queries = [
    "What are the latest guidelines for managing type 2 diabetes?",
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
    "Çölyak hastalığı tanı kriterleri nelerdir?",
]

articles = load_articles_from_json(JSON_PATH)
retriever = SemanticMedicalRetriever(articles)

for query in queries:
    print("=" * 100)
    print("QUERY:", query)
    print("=" * 100)

    results = retriever.search(query, top_k=5)

    for r in results:
        print(f"[{r.rank}] score={r.score:.4f} | PMID={r.pmid} | Year={r.year}")
        print(f"Title: {r.title}")
        print(f"Journal: {r.journal}")
        print(f"Matched terms: {r.matched_terms}")
        print(f"Preview: {r.retrieval_text_preview}")
        print("-" * 100)

    print()