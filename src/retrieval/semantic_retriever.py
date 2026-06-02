from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


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


class SemanticMedicalRetriever:
    def __init__(
        self,
        articles: List[Dict[str, Any]],
        model_name: str = MODEL_NAME,
    ) -> None:
        if not articles:
            raise ValueError("articles list is empty")

        self.articles = articles
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

        self.documents: List[str] = [self._build_document_text(article) for article in self.articles]
        self.prefixed_documents: List[str] = [
            f"{PASSAGE_PREFIX}{document}" for document in self.documents
        ]
        self.document_embeddings = self.model.encode(
            self.prefixed_documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _build_document_text(cls, article: Dict[str, Any]) -> str:
        title = cls._clean_text(article.get("title"))
        abstract = cls._clean_text(article.get("abstract"))

        if title and abstract:
            return f"{title} {abstract}"
        return title or abstract

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            return []

        normalized_query = query.strip()
        prefixed_query = f"{QUERY_PREFIX}{normalized_query}"

        query_embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = np.dot(self.document_embeddings, query_embedding)
        top_k = min(top_k, len(self.articles))
        ranked_indices = np.argsort(-scores)[:top_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            article = self.articles[int(idx)]
            retrieval_text = self.documents[int(idx)]

            results.append(
                SearchResult(
                    rank=rank,
                    score=float(scores[int(idx)]),
                    pmid=article.get("pmid"),
                    title=self._clean_text(article.get("title")),
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
        normalized_query = query.strip()
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "prefixed_query": f"{QUERY_PREFIX}{normalized_query}",
            "model_name": self.model_name,
        }


SemanticRetriever = SemanticMedicalRetriever


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
