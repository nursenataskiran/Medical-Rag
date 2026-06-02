from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

TR_EN_MEDICAL_MAP = {
    "çocuklarda": "pediatric",
    "akut otitis media": "acute otitis media",
    "tedavisi": "treatment",
    "tedavi": "treatment",
    "nasıl yapılır": "management",
    "çölyak hastalığı": "celiac disease",
    "tanı kriterleri": "diagnostic criteria",
    "gebelikte": "during pregnancy",
    "gebelik": "pregnancy",
    "demir eksikliği anemisi": "iron deficiency anemia",
}


STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "for", "of", "to", "in", "on", "with",
    "how", "what", "are", "is", "was", "were", "be", "during", "latest"
}


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


class BM25MedicalRetriever:
    """
    BM25 retriever over medical articles using title + abstract.
    Includes lightweight Turkish -> English medical query normalization.
    """

    def __init__(
        self,
        articles: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not articles:
            raise ValueError("articles list is empty")

        self.articles = articles
        self.k1 = k1
        self.b = b

        self.documents: List[str] = [self._build_document_text(a) for a in self.articles]
        self.tokenized_documents: List[List[str]] = [self.tokenize(doc) for doc in self.documents]

        self.bm25 = BM25Okapi(self.tokenized_documents, k1=self.k1, b=self.b)

    @staticmethod
    def _build_document_text(article: Dict[str, Any]) -> str:
        title = (article.get("title") or "").strip()
        abstract = (article.get("abstract") or "").strip()
        return f"{title} {abstract}".strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)?", text)
        return [t for t in tokens if t and t not in STOPWORDS_EN]

    @staticmethod
    def normalize_turkish_query(query: str) -> str:
        q = query.lower().strip()

        # longest phrases first to avoid partial replacements breaking bigger phrases
        for tr_phrase in sorted(TR_EN_MEDICAL_MAP.keys(), key=len, reverse=True):
            en_phrase = TR_EN_MEDICAL_MAP[tr_phrase]
            q = re.sub(rf"\b{re.escape(tr_phrase)}\b", en_phrase, q)

        # remove Turkish punctuation / leftovers
        q = re.sub(r"[^\w\s-]", " ", q, flags=re.UNICODE)
        q = re.sub(r"\s+", " ", q).strip()

        return q

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        normalized_query = self.normalize_turkish_query(query)
        query_tokens = self.tokenize(normalized_query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            article = self.articles[idx]
            retrieval_text = self.documents[idx]

            results.append(
                SearchResult(
                    rank=rank,
                    score=float(scores[idx]),
                    pmid=article.get("pmid"),
                    title=article.get("title", ""),
                    journal=article.get("journal"),
                    year=article.get("year"),
                    doi=article.get("doi"),
                    matched_terms=article.get("matched_terms", []),
                    normalized_query=normalized_query,
                    retrieval_text_preview=retrieval_text[:300].strip(),
                )
            )

        return results

    def explain_query(self, query: str) -> Dict[str, Any]:
        normalized_query = self.normalize_turkish_query(query)
        tokens = self.tokenize(normalized_query)
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "tokens": tokens,
            "k1": self.k1,
            "b": self.b,
        }


def load_articles_from_json(json_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "articles" in data and isinstance(data["articles"], list):
            articles = data["articles"]
        else:
            raise ValueError("JSON dict must contain an 'articles' list")
    elif isinstance(data, list):
        articles = data
    else:
        raise ValueError("JSON must be a list of articles or a dict with 'articles' key")

    required_any_of = {"title", "abstract"}
    for i, article in enumerate(articles):
        if not isinstance(article, dict):
            raise ValueError(f"Article at index {i} is not a dict")
        if not required_any_of.intersection(article.keys()):
            raise ValueError(
                f"Article at index {i} must contain at least one of: {required_any_of}"
            )

    return articles