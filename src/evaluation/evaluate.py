from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.retrieval.bm25_retriever import BM25MedicalRetriever, load_articles_from_json
from src.retrieval.semantic_retriever import SemanticMedicalRetriever
from src.retrieval.hybrid_retriever import HybridMedicalRetriever


DATA_PATH = Path("data/raw/pubmed_articles.json")
LABELS_PATH = Path("data/eval/evaluation_pool_labeled.csv")
TOP_K = 5

QUERIES = [
    "What are the latest guidelines for managing type 2 diabetes?",
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
    "Iron supplementation dosing for anemia during pregnancy",
    "Çölyak hastalığı tanı kriterleri nelerdir?",
    "Antibiotic resistance patterns in community acquired pneumonia",
]


def load_qrels(csv_path: Path) -> Dict[str, Dict[str, int]]:
    df = pd.read_csv(csv_path, sep=None, engine="python")

    required_cols = {"query", "pmid", "relevance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in labeled CSV: {missing}")

    df["query"] = df["query"].astype(str)
    df["pmid"] = df["pmid"].astype(str)
    df["relevance"] = df["relevance"].fillna(0).astype(float).astype(int)

    qrels: Dict[str, Dict[str, int]] = {}
    for query, group in df.groupby("query"):
        qrels[query] = {
            str(row["pmid"]): int(row["relevance"])
            for _, row in group.iterrows()
            if str(row["pmid"]).strip()
        }

    return qrels


def get_ranked_pmids(results: List[Any], top_k: int) -> List[str]:
    ranked = []
    for r in results[:top_k]:
        if r.pmid is None:
            continue
        ranked.append(str(r.pmid))
    return ranked


def precision_at_k(ranked_pmids: List[str], qrel: Dict[str, int], k: int = 5) -> float:
    if k == 0:
        return 0.0
    rels = [1 if qrel.get(pmid, 0) >= 1 else 0 for pmid in ranked_pmids[:k]]
    return sum(rels) / k


def reciprocal_rank(ranked_pmids: List[str], qrel: Dict[str, int], min_relevance: int = 2) -> float:
    for rank, pmid in enumerate(ranked_pmids, start=1):
        if qrel.get(pmid, 0) >= min_relevance:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_pmids: List[str], qrel: Dict[str, int], k: int = 5) -> float:
    dcg = 0.0
    for i, pmid in enumerate(ranked_pmids[:k], start=1):
        rel = qrel.get(pmid, 0)
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_pmids: List[str], qrel: Dict[str, int], k: int = 5) -> float:
    dcg = dcg_at_k(ranked_pmids, qrel, k)

    ideal_rels = sorted(qrel.values(), reverse=True)[:k]
    ideal_dcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        ideal_dcg += (2**rel - 1) / math.log2(i + 1)

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def evaluate_method(
    method_name: str,
    rankings: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 5,
) -> pd.DataFrame:
    rows = []

    for query in QUERIES:
        ranked_pmids = rankings[query]
        qrel = qrels.get(query, {})

        rows.append(
            {
                "method": method_name,
                "query": query,
                "P@5": precision_at_k(ranked_pmids, qrel, k),
                "MRR": reciprocal_rank(ranked_pmids, qrel, min_relevance=2),
                "nDCG@5": ndcg_at_k(ranked_pmids, qrel, k),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    articles = load_articles_from_json(DATA_PATH)
    qrels = load_qrels(LABELS_PATH)

    bm25 = BM25MedicalRetriever(
        articles=articles,
        k1=1.5,
        b=0.5,
    )
    semantic = SemanticMedicalRetriever(articles)
    hybrid = HybridMedicalRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        k=20,
    )

    bm25_rankings: Dict[str, List[str]] = {}
    semantic_rankings: Dict[str, List[str]] = {}
    hybrid_rankings: Dict[str, List[str]] = {}

    for query in QUERIES:
        bm25_rankings[query] = get_ranked_pmids(bm25.search(query, top_k=TOP_K), TOP_K)
        semantic_rankings[query] = get_ranked_pmids(semantic.search(query, top_k=TOP_K), TOP_K)
        hybrid_rankings[query] = get_ranked_pmids(hybrid.search(query, top_k=TOP_K), TOP_K)

    df_bm25 = evaluate_method("BM25", bm25_rankings, qrels, k=TOP_K)
    df_semantic = evaluate_method("Semantic", semantic_rankings, qrels, k=TOP_K)
    df_hybrid = evaluate_method("Hybrid (RRF)", hybrid_rankings, qrels, k=TOP_K)

    per_query_df = pd.concat([df_bm25, df_semantic, df_hybrid], ignore_index=True)

    summary_df = (
        per_query_df.groupby("method")[["P@5", "MRR", "nDCG@5"]]
        .mean()
        .reset_index()
        .sort_values(by="nDCG@5", ascending=False)
    )

    per_query_df[["P@5", "MRR", "nDCG@5"]] = per_query_df[["P@5", "MRR", "nDCG@5"]].round(4)
    summary_df[["P@5", "MRR", "nDCG@5"]] = summary_df[["P@5", "MRR", "nDCG@5"]].round(4)

    print("\n=== Overall Results ===")
    print(summary_df.to_string(index=False))

    print("\n=== Per-Query Results ===")
    print(per_query_df.to_string(index=False))


if __name__ == "__main__":
    main()