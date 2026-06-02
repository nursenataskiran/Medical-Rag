import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pubmed_client import PubMedClient  # noqa: E402

INPUT_CSV = PROJECT_ROOT / "data" / "input" / "medical_terms.csv"
OUTPUT_JSON = PROJECT_ROOT / "data" / "raw" / "pubmed_articles.json"


def read_terms(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)

    if "term" not in df.columns:
        raise ValueError(f"'term' column not found in {csv_path}")

    return [str(term).strip() for term in df["term"].dropna() if str(term).strip()]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    terms = read_terms(INPUT_CSV)
    client = PubMedClient(max_results=5)

    seen_articles: dict[str, dict] = {}
    total_fetched = 0
    duplicates_removed = 0
    errors = 0

    for term in terms:
        logging.info("Processing term: %s", term)

        pmids = client.search(term)
        if not pmids:
            errors += 1
            logging.warning("No PMIDs found for: %s", term)
            continue

        articles = client.fetch(pmids)
        if not articles:
            errors += 1
            logging.warning("No articles fetched for: %s", term)
            continue

        total_fetched += len(articles)

        for article in articles:
            if article.pmid in seen_articles:
                if term not in seen_articles[article.pmid]["matched_terms"]:
                    seen_articles[article.pmid]["matched_terms"].append(term)
                duplicates_removed += 1
                continue

            record = article.to_dict()
            record["matched_terms"] = [term]
            record["retrieval_text"] = f"{article.title} {article.abstract}".strip()
            seen_articles[article.pmid] = record

    records = list(seen_articles.values())

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("PubMed Data Pipeline Summary")
    print("=" * 50)
    print(f"Terms processed    : {len(terms)}")
    print(f"Total fetched      : {total_fetched}")
    print(f"Unique articles    : {len(records)}")
    print(f"Duplicates removed : {duplicates_removed}")
    print(f"Errors encountered : {errors}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()