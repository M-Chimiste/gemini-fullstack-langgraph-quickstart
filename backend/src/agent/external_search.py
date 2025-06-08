import time
import random
from typing import List, Dict, Optional
import requests


class SemanticScholarSearch:
    """
    A class for performing semantic searches on the Semantic Scholar API.

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
        """
        Performs a search on Semantic Scholar using the provided query and parameters.

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

            except (requests.exceptions.RetryError, requests.exceptions.RequestException) as e:
                if retries >= self.max_retries:
                    raise RuntimeError(f"Semantic Scholar API request failed: {e}")

                # Exponential back-off with jitter
                sleep_for = self.backoff_factor * (2 ** retries) + random.uniform(
                    0, 1
                )
                time.sleep(sleep_for)
                retries += 1

    @staticmethod
    def _normalize_paper(paper: Dict) -> Dict:
        """
        Normalize a paper dictionary by flattening its structure and adding a 'pdf_url' field if available.

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