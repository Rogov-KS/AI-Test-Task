from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Protocol, TypedDict

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.core import settings

logger = logging.getLogger(__name__)


class KnowledgeProvider(Protocol):
    async def get_entity_page(self, entity: str) -> "WikipediaPage": ...
    async def get_entity_pages(self, entity: str, *, limit: int = 3) -> list["WikipediaPage"]: ...


class WikipediaPage(TypedDict):
    query: str
    title: str
    extract: str


class WikipediaLookupError(RuntimeError):
    pass


class WikipediaKnowledgeProvider:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        api_url: str = settings.wikipedia_api_url,
        timeout_s: float = settings.wikipedia_timeout_s,
        user_agent: str = settings.wikipedia_user_agent,
    ) -> None:
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._api_url = api_url
        self._timeout_s = timeout_s
        self._user_agent = user_agent

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        retry=retry_if_exception_type((httpx.HTTPError, WikipediaLookupError)),
        reraise=True,
    )
    async def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.get(
            self._api_url,
            params=params,
            headers={"User-Agent": self._user_agent},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise WikipediaLookupError("wikipedia returned unexpected payload")
        return payload

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        return {token for token in re.findall(r"\w+", value.lower()) if token}

    def _score_search_hit(self, query: str, hit: dict[str, Any]) -> float:
        title = hit.get("title")
        if not isinstance(title, str):
            return float("-inf")
        title_normalized = title.strip().lower()
        query_normalized = query.strip().lower()
        query_tokens = self._tokenize(query_normalized)
        title_tokens = self._tokenize(title_normalized)

        score = 0.0
        if title_normalized == query_normalized:
            score += 100.0
        if query_normalized in title_normalized:
            score += 25.0
        if title_normalized in query_normalized:
            score += 10.0
        if query_tokens and title_tokens:
            overlap = len(query_tokens.intersection(title_tokens))
            score += overlap * 8.0
            score += overlap / len(query_tokens)

        wordcount = hit.get("wordcount")
        if isinstance(wordcount, int) and wordcount > 0:
            score += min(wordcount / 1000.0, 2.0)
        return score

    async def _search_titles(self, query: str, *, limit: int = 1) -> list[str]:
        payload = await self._request(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "utf8": 1,
            }
        )
        search = payload.get("query", {}).get("search", [])
        if not search:
            raise WikipediaLookupError(f"no wikipedia page found for {query!r}")

        sorted_hits = sorted(
            (hit for hit in search if isinstance(hit, dict)),
            key=lambda hit: self._score_search_hit(query, hit),
            reverse=True,
        )
        if not sorted_hits:
            raise WikipediaLookupError("wikipedia search response is malformed")
        titles: list[str] = []
        for hit in sorted_hits:
            title = hit.get("title")
            if isinstance(title, str) and title not in titles:
                titles.append(title)
            if len(titles) >= max(1, limit):
                break
        if not titles:
            raise WikipediaLookupError("wikipedia title is malformed")
        return titles

    async def _load_extract(self, title: str) -> str:
        payload = await self._request(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "titles": title,
                "utf8": 1,
            }
        )
        pages = payload.get("query", {}).get("pages", {})
        if not isinstance(pages, dict) or not pages:
            raise WikipediaLookupError(f"no wikipedia extract found for {title!r}")
        page = next(iter(pages.values()))
        if not isinstance(page, dict):
            raise WikipediaLookupError("wikipedia page response is malformed")
        extract = page.get("extract")
        if not isinstance(extract, str) or not extract.strip():
            raise WikipediaLookupError(f"wikipedia extract for {title!r} is empty")
        return extract

    async def get_entity_page(self, entity: str) -> WikipediaPage:
        pages = await self.get_entity_pages(entity, limit=1)
        return pages[0]

    async def get_entity_pages(self, entity: str, *, limit: int = 3) -> list[WikipediaPage]:
        logger.info("wikipedia get_entity_page called", extra={"entity": entity})
        titles = await self._search_titles(entity, limit=limit)
        extracts = await asyncio.gather(*(self._load_extract(title) for title in titles))
        return [
            {"query": entity, "title": title, "extract": extract}
            for title, extract in zip(titles, extracts, strict=True)
        ]
