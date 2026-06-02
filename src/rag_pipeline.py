from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.retrieval.bm25_retriever import BM25MedicalRetriever
from src.retrieval.semantic_retriever import (
    SemanticMedicalRetriever,
    load_articles_from_json,
)
from src.retrieval.hybrid_retriever import HybridMedicalRetriever
load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """
You are a medical QA assistant.

Answer ONLY from the provided context.
Do not use outside knowledge.
If the context does not support the answer, say:
"Sağlanan bağlama dayanarak bu soruyu güvenle yanıtlayamıyorum."

Rules:
- Always answer in Turkish.
- Do not translate medical terms, disease names, drug names, procedure names, guideline names, article titles, or citation text unless a Turkish equivalent is explicitly provided in the context.
- Preserve important technical terminology in its original form when appropriate.
- Be concise and clinically clear.
- Cite sources by PMID or article title.
- Never invent citations.
- If evidence is limited or indirect, clearly state that.
""".strip()

@dataclass
class RetrievedDocument:
    pmid: str
    title: str
    journal: str
    year: str
    doi: str
    abstract: str


class MedicalRAGPipeline:
    def __init__(self, articles_path: str | Path) -> None:
        self.articles_path = Path(articles_path)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Please define it in your .env file."
            )

        self.articles = load_articles_from_json(self.articles_path)

        self.article_by_pmid = {
            str(article.get("pmid")): article
            for article in self.articles
            if article.get("pmid") is not None
        }

        bm25 = BM25MedicalRetriever(self.articles)
        semantic = SemanticMedicalRetriever(self.articles)

        self.retriever = HybridMedicalRetriever(bm25, semantic)
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _to_full_document(self, result: Any) -> RetrievedDocument:
        pmid = self._clean(result.pmid)
        article = self.article_by_pmid.get(pmid, {})

        return RetrievedDocument(
            pmid=pmid,
            title=self._clean(article.get("title") or result.title),
            journal=self._clean(article.get("journal") or result.journal),
            year=self._clean(article.get("year") or result.year),
            doi=self._clean(article.get("doi") or result.doi),
            abstract=self._clean(article.get("abstract")) or self._clean(result.retrieval_text_preview),
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        results = self.retriever.search(query, top_k=top_k)
        return [self._to_full_document(r) for r in results]

    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        chunks: List[str] = []

        for i, doc in enumerate(documents, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"[Document {i}]",
                        f"PMID: {doc.pmid or 'N/A'}",
                        f"Title: {doc.title or 'N/A'}",
                        f"Journal: {doc.journal or 'N/A'}",
                        f"Year: {doc.year or 'N/A'}",
                        f"DOI: {doc.doi or 'N/A'}",
                        f"Abstract: {doc.abstract or 'N/A'}",
                    ]
                )
            )

        return "\n\n".join(chunks)

    def _build_user_prompt(self, query: str, documents: List[RetrievedDocument]) -> str:
        context = self._build_context(documents)

        return f"""
Question:
{query}

Context:
{context}

Instructions:
- Answer the question using only the context above.
- Always write the final answer in Turkish.
- Do not translate core medical terminology unless clearly supported by the context.
- If the context is incomplete, clearly state what is supported and what is missing.
- Cite supporting sources inline using PMID or article title.
""".strip()

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        documents = self.retrieve(query=query, top_k=top_k)
        user_prompt = self._build_user_prompt(query, documents)

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )

        return {
            "query": query,
            "model": MODEL_NAME,
            "retrieved_documents": documents,
            "answer": (response.text or "").strip(),
        }