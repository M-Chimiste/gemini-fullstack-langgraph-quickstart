"""Utilities for searching papers using Semantic Scholar."""

import os
import random
import tempfile
import time
from typing import Dict, List, Optional

import requests

from ..pdf.parsers import FlatMarkdownParser
from ..pdf.processing import SpacyLayoutDocProcessor
from .local_search import BaseSearchTool


class SemanticScholarSearch:
    """A class for performing semantic searches on the Semantic Scholar API.

    This class provides a convenient interface for searching papers on the Semantic Scholar API. It handles the construction of the search query, sending the request, and parsing the response. The search results are returned as a list of dictionaries, each representing a paper and its details.

    Attributes:
        BASE_URL (str): The base URL for the Semantic Scholar API.
        max_retries (int): The maximum number of retries for failed requests.
        backoff_factor (float): The factor to use for exponential backoff in case of retries.
        session (requests.Session): The session to use for requests.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        session: Optional[requests.Session] = None,
    ):
        """Initialize the searcher."""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        # Re-use a session for connection pooling
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                # Omitting a custom User-Agent avoids the 4xx rejection you saw
                "Accept": "application/json",
            }
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        require_pdf: bool = True,
        fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Perform a search on Semantic Scholar using the provided query and parameters.

        Args:
            query (str): The search query to execute.
            limit (int, optional): The maximum number of results to return. Defaults to 10.
            offset (int, optional): The offset for pagination. Defaults to 0.
            require_pdf (bool, optional): If True, only returns papers with a PDF available. Defaults to True.
            fields (Optional[List[str]], optional): A list of fields to include in the response. Defaults to None.

        Returns:
            List[Dict]: A list of dictionaries, each representing a paper and its details.
        """
        if fields is None:
            fields = [
                "title",
                "year",
                "authors",
                "abstract",
                "url",
                "isOpenAccess",
                "openAccessPdf",
            ]

        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": ",".join(fields),
        }

        data = self._request(params)

        papers = [
            self._normalize_paper(p)
            for p in data.get("data", [])
            if not require_pdf  # no filtering
            or (p.get("openAccessPdf") and p["openAccessPdf"].get("url"))
        ]

        return papers

    def _request(self, params: Dict) -> Dict:
        """GET with exponential back-off on 429/5xx and network errors."""
        retries = 0
        while True:
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=20)
                if resp.status_code == 429:
                    raise requests.exceptions.RetryError(
                        "Rate limit reached (429)", response=resp
                    )
                resp.raise_for_status()
                return resp.json()

            except (
                requests.exceptions.RetryError,
                requests.exceptions.RequestException,
            ) as e:
                if retries >= self.max_retries:
                    raise RuntimeError(f"Semantic Scholar API request failed: {e}")

                # Exponential back-off with jitter
                sleep_for = self.backoff_factor * (2**retries) + random.uniform(0, 1)
                time.sleep(sleep_for)
                retries += 1

    @staticmethod
    def _normalize_paper(paper: Dict) -> Dict:
        """Normalize a paper dictionary by flattening its structure and adding a 'pdf_url' field if available.

        This method takes a paper dictionary as input, which may contain nested structures. It flattens the dictionary
        by removing any nested structures and adding the 'pdf_url' field if the paper has an open access PDF available.
        The resulting dictionary is a shallow copy of the original, ensuring the original data structure is not modified.

        Args:
            paper (Dict): The paper dictionary to be normalized.

        Returns:
            Dict: A flattened dictionary representation of the paper with an added 'pdf_url' field if applicable.
        """
        pdf_url = None
        if paper.get("openAccessPdf"):
            pdf_url = paper["openAccessPdf"].get("url")

        # Make a shallow copy so we don’t mutate the original dict
        flat = {k: v for k, v in paper.items() if k != "openAccessPdf"}
        if pdf_url:
            flat["pdf_url"] = pdf_url
        return flat


class ExternalSearchTool(BaseSearchTool):
    """Search tool that queries Semantic Scholar and optionally downloads PDFs."""

    def __init__(self, enable_pdf_download: bool = True):
        """Create a new tool instance."""
        self.searcher = SemanticScholarSearch()
        self.enable_pdf_download = enable_pdf_download
        self.pdf_processor = SpacyLayoutDocProcessor(
            language="en",
            save_text=False,
            export_tables=False,
            export_figures=False,
            remove_md_image_tags=True,
        )

    def find_papers_by_str(self, query: str, limit: int = 10) -> str:
        """Search Semantic Scholar and return a formatted summary."""
        try:
            papers = self.searcher.search(query, limit=limit)
        except Exception as e:  # pragma: no cover - network failure handling
            return f"External search failed: {e}"

        if not papers:
            return f"No papers found for query: '{query}'"

        formatted = []
        for paper in papers:
            info = (
                f"Title: {paper.get('title')}\n"
                f"Year: {paper.get('year')}\n"
                f"URL: {paper.get('url')}\n"
            )
            if paper.get("pdf_url"):
                info += f"PDF: {paper['pdf_url']}\n"
            if paper.get("abstract"):
                info += f"Abstract: {paper['abstract'][:300]}...\n"
            formatted.append(info + "---")

        return f"Found {len(papers)} papers for query: '{query}'\n\n" + "\n".join(
            formatted
        )

    def retrieve_full_text(self, paper_id: str) -> str:
        """Download and process a PDF from the provided URL."""
        if not self.enable_pdf_download:
            return "PDF download disabled."
        return self._download_and_process_pdf(paper_id)

    def _download_and_process_pdf(self, url: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                tmp.write(response.content)
                tmp_path = tmp.name

            try:
                result = self.pdf_processor.process_document(tmp_path)
                markdown_text = result.get("processed_data", "")
                parser = FlatMarkdownParser(markdown_text, max_tokens=60000)
                parsed_chunks = parser.get_parsed_data()
                return "\n".join(parsed_chunks)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:  # pragma: no cover - network failure handling
            return f"Error processing PDF from {url}: {e}"
