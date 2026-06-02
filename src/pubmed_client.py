from dataclasses import dataclass
import logging
import time
from typing import Any

import requests
from lxml import etree

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    first_author: str
    journal: str
    year: str
    doi: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "first_author": self.first_author,
            "journal": self.journal,
            "year": self.year,
            "doi": self.doi,
        }


class PubMedClient:
    def __init__(
        self,
        max_results: int = 5,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_requests_per_sec: float = 2.5,
    ) -> None:
        self.max_results = max_results
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.min_interval = 1.0 / max_requests_per_sec
        self.session = requests.Session()
        self.last_request_time = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.monotonic()

    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        for attempt in range(1, self.max_retries + 1):
            try:
                self._throttle()
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                wait = self.backoff_factor * (2 ** (attempt - 1))
                logger.warning(
                    "Request failed (%s) attempt %s/%s. Retrying in %.1fs",
                    exc,
                    attempt,
                    self.max_retries,
                    wait,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(wait)

        raise RuntimeError("Unexpected request retry failure")

    def search(self, term: str) -> list[str]:
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": self.max_results,
            "sort": "date",
            "retmode": "xml",
        }

        try:
            response = self._get(ESEARCH_URL, params)
        except requests.RequestException:
            logger.error("esearch failed for term: %s", term)
            return []

        root = etree.fromstring(response.content)
        pmids = [node.text for node in root.findall(".//Id") if node.text]
        return pmids

    def fetch(self, pmids: list[str]) -> list[PubMedArticle]:
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        try:
            response = self._get(EFETCH_URL, params)
        except requests.RequestException:
            logger.error("efetch failed for pmids: %s", pmids)
            return []

        return self._parse_articles(response.content)

    @staticmethod
    def _safe_text(element, xpath: str, default: str = "") -> str:
        if element is None:
            return default
        node = element.find(xpath)
        if node is None or node.text is None:
            return default
        return node.text.strip()

    @staticmethod
    def _parse_articles(xml_bytes: bytes) -> list[PubMedArticle]:
        root = etree.fromstring(xml_bytes)
        articles: list[PubMedArticle] = []

        for article_el in root.findall(".//PubmedArticle"):
            try:
                medline = article_el.find(".//MedlineCitation")
                article = medline.find(".//Article") if medline is not None else None

                pmid = PubMedClient._safe_text(medline, "PMID")
                if not pmid:
                    continue

                title = PubMedClient._safe_text(article, "ArticleTitle")

                abstract_parts: list[str] = []
                if article is not None:
                    for abstract_node in article.findall(".//AbstractText"):
                        text = "".join(abstract_node.itertext()).strip()
                        label = abstract_node.get("Label", "").strip()
                        if text:
                            abstract_parts.append(f"{label}: {text}" if label else text)
                abstract = " ".join(abstract_parts)

                first_author = ""
                if article is not None:
                    authors = article.findall(".//Author")
                    if authors:
                        last_name = PubMedClient._safe_text(authors[0], "LastName")
                        fore_name = PubMedClient._safe_text(authors[0], "ForeName")
                        first_author = f"{last_name} {fore_name}".strip()

                journal = PubMedClient._safe_text(article, ".//Journal/Title")

                year = ""
                for xpath in [
                    ".//ArticleDate/Year",
                    ".//Journal/JournalIssue/PubDate/Year",
                    ".//Journal/JournalIssue/PubDate/MedlineDate",
                ]:
                    year = PubMedClient._safe_text(article, xpath)
                    if year:
                        year = year[:4]
                        break

                doi = ""
                if article is not None:
                    for node in article.findall(".//ELocationID"):
                        if node.get("EIdType") == "doi" and node.text:
                            doi = node.text.strip()
                            break

                articles.append(
                    PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        first_author=first_author,
                        journal=journal,
                        year=year,
                        doi=doi,
                    )
                )
            except Exception:
                logger.exception("Failed to parse one PubMed article")

        return articles