import asyncio
import os
import uuid
from html.parser import HTMLParser
from typing import Any

import httpx
from fastmcp import Context, FastMCP
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC  # noqa: N812 — Selenium convention
from selenium.webdriver.support.ui import WebDriverWait

from ...logger_config import get_logger
from ..state_manager import (
    create_session_if_not_exists,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Primary search: DuckDuckGo Lite via httpx (no browser required)
# ---------------------------------------------------------------------------


def _search_duckduckgo_lite(query: str, num_results: int, timeout: int) -> list[dict[str, str]]:
    """Fetch search results from DuckDuckGo Lite without launching a browser.

    DuckDuckGo Lite returns plain HTML with no JavaScript, making it immune
    to bot-detection that blocks Selenium/Chromium on Google.
    """

    class _DDGLiteParser(HTMLParser):
        """Extract result-link / result-snippet pairs from DDG Lite HTML."""

        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.results: list[dict[str, str]] = []
            self._pending: dict[str, str] = {}
            self._in_link = False
            self._in_snippet = False

        def handle_starttag(self, tag: str, attrs: list) -> None:  # type: ignore[override]
            ad = dict(attrs)
            cls = ad.get("class", "")
            if tag == "a" and "result-link" in cls:
                self._pending = {"link": ad.get("href", ""), "title": "", "snippet": ""}
                self._in_link = True
            elif tag == "td" and "result-snippet" in cls:
                self._in_snippet = True

        def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
            if tag == "a" and self._in_link:
                self._in_link = False
            elif tag == "td" and self._in_snippet:
                self._in_snippet = False
                if self._pending.get("title") and self._pending.get("link"):
                    self.results.append(self._pending.copy())
                    self._pending = {}

        def handle_data(self, data: str) -> None:  # type: ignore[override]
            stripped = data.strip()
            if not stripped:
                return
            if self._in_link:
                self._pending["title"] = self._pending.get("title", "") + stripped
            elif self._in_snippet:
                self._pending["snippet"] = self._pending.get("snippet", "") + stripped

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept": "text/html,application/xhtml+xml",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers=headers,
        )
    resp.raise_for_status()
    parser = _DDGLiteParser()
    parser.feed(resp.text)
    return parser.results[:num_results]


# ---------------------------------------------------------------------------
# Fallback search: Selenium + DuckDuckGo (browser-based)
# ---------------------------------------------------------------------------


def _get_chromedriver_service() -> ChromeService:
    """Return a ChromeService using the system chromedriver when available.

    The Docker image installs chromium + chromium-driver together from apt, so
    they are always version-matched.  Prefer that over webdriver-manager which
    would download a stale pinned driver version.
    """
    system_driver = os.environ.get("CHROMEDRIVER_PATH", "")
    if system_driver and os.path.isfile(system_driver):
        logger.info("Using system ChromeDriver: %s", system_driver)
        return ChromeService(executable_path=system_driver)

    try:
        from webdriver_manager.chrome import ChromeDriverManager  # noqa: PLC0415

        logger.info("CHROMEDRIVER_PATH not set; using webdriver_manager to locate driver.")
        return ChromeService(ChromeDriverManager().install())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not locate a ChromeDriver.  Set CHROMEDRIVER_PATH to the "
            "chromedriver binary or install webdriver-manager."
        ) from exc


async def ensure_session_initialized_websearch(ctx: Context, override_session_id: str | None = None) -> str:
    """Helper to ensure the session exists in the database."""
    session_id = override_session_id or get_stable_session_id_websearch(ctx)
    client_id = ctx.client_id if hasattr(ctx, "client_id") else None
    await create_session_if_not_exists(session_id, client_id)
    return session_id


def get_stable_session_id_websearch(ctx: Context) -> str:
    """Retrieves or generates a stable session ID for the websearch tool context."""
    if hasattr(ctx, "session_id") and ctx.session_id:
        return ctx.session_id
    if ctx.request_id:
        return ctx.request_id
    logger.warning("WebSearch Tool: Could not determine a stable session ID. Generating a new one.")
    return str(uuid.uuid4())


def _run_selenium_search(query: str, num_results: int, wait_timeout: int) -> list[dict[str, str]]:
    """Browser-based fallback using DuckDuckGo (not Google, which blocks headless bots)."""
    results = []
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("window-size=1920x1080")

    chrome_bin = os.environ.get("CHROME_BIN", "")
    if chrome_bin:
        chrome_options.binary_location = chrome_bin

    driver = None
    try:
        service = _get_chromedriver_service()
        logger.info("ChromeDriver service initialized (Selenium fallback).")

        driver = webdriver.Chrome(service=service, options=chrome_options)
        # Use DuckDuckGo — it allows headless browsers unlike Google
        ddg_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        logger.info(f"WebDriver navigating to: {ddg_url}")
        driver.get(ddg_url)

        logger.info("Waiting for DuckDuckGo search results (.result__title)...")
        WebDriverWait(driver, wait_timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".result__title")))
        logger.info("Search results container found.")

        entries = driver.find_elements(By.CSS_SELECTOR, ".result")[:num_results]
        logger.info(f"Found {len(entries)} DuckDuckGo result elements.")

        for i, entry in enumerate(entries):
            try:
                title_el = entry.find_element(By.CSS_SELECTOR, ".result__a")
                link = title_el.get_attribute("href") or ""
                title = title_el.text
                snippet_els = entry.find_elements(By.CSS_SELECTOR, ".result__snippet")
                snippet = snippet_els[0].text if snippet_els else "N/A"
                if title and link:
                    results.append({"title": title, "link": link, "snippet": snippet})
            except Exception as e:
                logger.warning(f"Could not parse DDG result entry {i + 1}: {type(e).__name__} - {e}")
                continue
    except TimeoutException as te:
        logger.error(
            f"Selenium search timed out after {wait_timeout} seconds waiting for results for query: '{query}'",
            exc_info=True,
        )
        raise RuntimeError(
            f"Selenium timed out after {wait_timeout}s waiting for search results for query '{query}'. Original error: {te}"
        ) from te
    except Exception as e:
        logger.error(f"Selenium search failed: {type(e).__name__} - {e}", exc_info=True)
        raise RuntimeError(f"Selenium execution failed: {type(e).__name__} - {e}") from e
    finally:
        if driver:
            logger.info("Quitting WebDriver.")
            driver.quit()
    return results


def register_tools(mcp: FastMCP):
    @mcp.tool()
    async def perform_web_search(
        query: str,
        ctx: Context,
        num_results: int = 5,
        timeout: int = 90,
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Performs a web search and returns the top N results.

        Uses DuckDuckGo Lite via HTTP as the primary path (fast, no browser).
        Falls back to a Selenium-based DuckDuckGo search if the HTTP request fails.
        """
        logger.info(f"Web search tool called with query: '{query}', num_results: {num_results}, timeout: {timeout}")

        session_id = await ensure_session_initialized_websearch(ctx, override_session_id=mcp_session_id)
        await ctx.info(f"Session {session_id}: Performing web search for query: '{query}'")

        if not session_id:
            await ctx.error("Session ID is required for web search.")
            return {"error": "Session ID is required for web search."}

        try:
            logger.info("Attempting DuckDuckGo Lite search via httpx...")
            search_results = await asyncio.to_thread(_search_duckduckgo_lite, query, num_results, timeout)
            if search_results:
                await ctx.info(f"Found {len(search_results)} results for query: '{query}'")
                return {"query": query, "results": search_results}

            logger.warning("DDG Lite returned no results; falling back to Selenium.")
            search_results = await asyncio.to_thread(_run_selenium_search, query, num_results, timeout)

            if not search_results:
                await ctx.warning(f"No search results found for query: '{query}'")
                return {"query": query, "results": [], "message": "No results found."}

            await ctx.info(f"Found {len(search_results)} results (Selenium fallback) for query: '{query}'")
            return {"query": query, "results": search_results}
        except Exception as e:
            logger.error(f"Error during web search for query '{query}': {e}", exc_info=True)
            await ctx.error(f"Web search failed: {e}")
            return {"error": f"Failed to perform web search: {e}"}

    logger.info("Web search MCP tool registered.")
