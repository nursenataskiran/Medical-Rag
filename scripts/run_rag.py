from pathlib import Path

from src.rag_pipeline import MedicalRAGPipeline


DATA_PATH = Path("data/raw/pubmed_articles.json")


def print_result(result: dict) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {result['query']}")
    print("-" * 100)

    print("Retrieved documents:")
    for i, doc in enumerate(result["retrieved_documents"], start=1):
        print(f"[{i}] PMID={doc.pmid} | Year={doc.year} | Title={doc.title}")

    print("\nAnswer:")
    print(result["answer"])

    print("\n" + "=" * 100)
    print()


def main() -> None:
    pipeline = MedicalRAGPipeline(articles_path=DATA_PATH)

    print("Medical RAG demo")
    print("Type a medical question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Query> ").strip()

        if not query:
            print("Please enter a non-empty query.\n")
            continue

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        try:
            result = pipeline.answer(query=query, top_k=5)
            print_result(result)
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()