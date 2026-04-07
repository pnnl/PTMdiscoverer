from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pubchempy as pcp
import pytest
from selenium.common.exceptions import TimeoutException

from agentic_framework_pkg.mcp_server.tools import pubchem_tool, uniprot_tool, websearch_tool


class FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self):
        def decorator(func: Any) -> Any:
            self.tools[func.__name__] = func
            return func

        return decorator


class FakeCtx:
    def __init__(
        self,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
        client_id: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.request_id = request_id
        self.client_id = client_id
        self.infos: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    async def info(self, message: str) -> None:
        self.infos.append(message)

    async def error(self, message: str) -> None:
        self.errors.append(message)

    async def warning(self, message: str) -> None:
        self.warnings.append(message)


async def _immediate_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


class FakeAsyncResponse:
    def __init__(
        self, payload: Any = None, *, status_code: int = 200, text: str = "OK", json_exc: Exception | None = None
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self._json_exc = json_exc

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    def json(self) -> Any:
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


class FakeAsyncClient:
    def __init__(self, response: FakeAsyncResponse | Exception) -> None:
        self.response = response

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        return None

    async def get(self, url: str) -> FakeAsyncResponse:
        del url
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pubchem_tools_cover_success_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCP()
    ctx = FakeCtx()

    monkeypatch.setattr(pubchem_tool.asyncio, "to_thread", _immediate_to_thread)
    pubchem_tool.register_tools(cast(Any, fake_mcp))

    compounds = [
        SimpleNamespace(cid=1, synonyms=["aspirin", "acetylsalicylic acid"], iupac_name="aspirin"),
        SimpleNamespace(cid=2, synonyms=None, iupac_name=None),
    ]
    monkeypatch.setattr(pubchem_tool.pcp, "get_compounds", lambda *args, **kwargs: compounds)
    result = await fake_mcp.tools["search_pubchem_by_name"](ctx, "aspirin", max_results=1)
    assert result["results"][0]["cid"] == 1

    monkeypatch.setattr(pubchem_tool.pcp, "get_compounds", lambda *args, **kwargs: [])
    empty_result = await fake_mcp.tools["search_pubchem_by_query"](ctx, "unknown")
    assert empty_result["message"].startswith("No compounds")

    monkeypatch.setattr(
        pubchem_tool.pcp,
        "get_compounds",
        lambda *args, **kwargs: (_ for _ in ()).throw(pcp.PubChemPyError("api down")),
    )
    error_result = await fake_mcp.tools["search_pubchem_by_name"](ctx, "aspirin")
    assert "error" in error_result

    compound = SimpleNamespace(
        molecular_formula="C9H8O4",
        molecular_weight=180.16,
        canonical_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        synonyms=["one", "two", "three", "four", "five", "six"],
    )
    monkeypatch.setattr(pubchem_tool.pcp.Compound, "from_cid", lambda cid: compound)
    props = await fake_mcp.tools["get_pubchem_compound_properties"](
        ctx, 2244, properties=["molecular_formula", "synonyms", "missing"]
    )
    assert props["properties"]["molecular_formula"] == "C9H8O4"
    assert props["properties"]["synonyms"] == ["one", "two", "three", "four", "five"]
    assert props["properties"]["missing"] is None

    monkeypatch.setattr(pubchem_tool.pcp.Compound, "from_cid", lambda cid: None)
    not_found = await fake_mcp.tools["get_pubchem_compound_properties"](ctx, 999)
    assert not_found["message"].endswith("not found.")

    monkeypatch.setattr(
        pubchem_tool.pcp.Compound,
        "from_cid",
        lambda cid: (_ for _ in ()).throw(pcp.PubChemPyError("bad cid")),
    )
    prop_error = await fake_mcp.tools["get_pubchem_compound_properties"](ctx, 2244)
    assert "error" in prop_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uniprot_tools_cover_success_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCP()
    ctx = FakeCtx(request_id="req-1", client_id="client-1")
    created_sessions: list[tuple[str, str | None]] = []

    async def fake_create_session(
        session_id: str, client_id: str | None, initial_context: dict[str, Any] | None = None
    ) -> None:
        del initial_context
        created_sessions.append((session_id, client_id))

    monkeypatch.setattr(uniprot_tool, "create_session_if_not_exists", fake_create_session)
    uniprot_tool.register_tools(cast(Any, fake_mcp))

    success_payload = {
        "accession": "P12345",
        "id": "TEST_HUMAN",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Protein"}}},
        "genes": [{"geneName": {"value": "GENE1"}}],
        "organism": {"scientificName": "Homo sapiens"},
        "sequence": {"length": 100, "mass": 12000},
        "features": [{"type": "Domain"}, {"type": "Other"}],
        "comments": [{"type": "Function"}, {"type": "Ignore"}],
    }
    monkeypatch.setattr(
        uniprot_tool.httpx, "AsyncClient", lambda timeout=20.0: FakeAsyncClient(FakeAsyncResponse(success_payload))
    )
    accession_result = await fake_mcp.tools["query_uniprot_by_accession"]("P12345", ctx, mcp_session_id="override")
    assert accession_result["accession"] == "P12345"
    assert accession_result["featuresSummary"] == ["Domain"]
    assert created_sessions[0] == ("override", "client-1")

    request = httpx.Request("GET", "https://rest.uniprot.org")
    http_error = httpx.HTTPStatusError(
        "bad", request=request, response=httpx.Response(404, request=request, text="missing")
    )
    monkeypatch.setattr(uniprot_tool.httpx, "AsyncClient", lambda timeout=20.0: FakeAsyncClient(http_error))
    assert "error" in await fake_mcp.tools["query_uniprot_by_accession"]("P12345", ctx)

    monkeypatch.setattr(
        uniprot_tool.httpx,
        "AsyncClient",
        lambda timeout=20.0: FakeAsyncClient(httpx.RequestError("network", request=httpx.Request("GET", "https://x"))),
    )
    assert "error" in await fake_mcp.tools["query_uniprot_by_accession"]("P12345", ctx)

    json_error = json.JSONDecodeError("bad json", "doc", 0)
    monkeypatch.setattr(
        uniprot_tool.httpx, "AsyncClient", lambda timeout=20.0: FakeAsyncClient(FakeAsyncResponse(json_exc=json_error))
    )
    assert "error" in await fake_mcp.tools["query_uniprot_by_accession"]("P12345", ctx)

    gene_payload = {
        "results": [
            {
                "primaryAccession": "P1",
                "uniProtkbId": "ENTRY1",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Protein 1"}}},
                "genes": [{"geneName": {"value": "APP"}}],
                "organism": {"scientificName": "Homo sapiens"},
            }
        ]
    }
    monkeypatch.setattr(
        uniprot_tool.httpx, "AsyncClient", lambda timeout=30.0: FakeAsyncClient(FakeAsyncResponse(gene_payload))
    )
    gene_result = await fake_mcp.tools["query_uniprot_by_gene"]("APP", ctx)
    assert gene_result["results"][0]["geneNames"] == ["APP"]

    monkeypatch.setattr(
        uniprot_tool.httpx, "AsyncClient", lambda timeout=30.0: FakeAsyncClient(FakeAsyncResponse({"results": []}))
    )
    no_results = await fake_mcp.tools["query_uniprot_by_gene"]("MISSING", ctx)
    assert no_results["message"] == "No matching entries found."

    request = httpx.Request("GET", "https://rest.uniprot.org")
    http_error = httpx.HTTPStatusError(
        "bad", request=request, response=httpx.Response(500, request=request, text="boom")
    )
    monkeypatch.setattr(uniprot_tool.httpx, "AsyncClient", lambda timeout=30.0: FakeAsyncClient(http_error))
    assert "error" in await fake_mcp.tools["query_uniprot_by_gene"]("APP", ctx)

    monkeypatch.setattr(
        uniprot_tool.httpx, "AsyncClient", lambda timeout=30.0: FakeAsyncClient(RuntimeError("unexpected"))
    )
    assert "error" in await fake_mcp.tools["query_uniprot_by_gene"]("APP", ctx)


@pytest.mark.unit
def test_uniprot_and_websearch_session_id_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = FakeCtx(session_id="sid", request_id="rid")
    assert uniprot_tool.get_stable_session_id_uniprot(cast(Any, ctx)) == "sid"
    assert websearch_tool.get_stable_session_id_websearch(cast(Any, ctx)) == "sid"

    ctx = FakeCtx(request_id="rid")
    assert uniprot_tool.get_stable_session_id_uniprot(cast(Any, ctx)) == "rid"
    assert websearch_tool.get_stable_session_id_websearch(cast(Any, ctx)) == "rid"

    monkeypatch.setattr(uniprot_tool.uuid, "uuid4", lambda: "generated-id")
    assert uniprot_tool.get_stable_session_id_uniprot(cast(Any, FakeCtx())) == "generated-id"
    assert websearch_tool.get_stable_session_id_websearch(cast(Any, FakeCtx())) == "generated-id"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websearch_tool_helpers_and_registered_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    del tmp_path
    fake_mcp = FakeMCP()
    ctx = FakeCtx(request_id="req-2", client_id="client-2")
    created_sessions: list[tuple[str, str | None]] = []

    async def fake_create_session(
        session_id: str, client_id: str | None, initial_context: dict[str, Any] | None = None
    ) -> None:
        del initial_context
        created_sessions.append((session_id, client_id))

    monkeypatch.setattr(websearch_tool, "create_session_if_not_exists", fake_create_session)
    monkeypatch.setattr(websearch_tool.asyncio, "to_thread", _immediate_to_thread)

    class FakeSyncResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class FakeSyncClient:
        def __init__(self, timeout: int, follow_redirects: bool) -> None:
            del timeout, follow_redirects

        def __enter__(self) -> FakeSyncClient:
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            del exc_type, exc, tb
            return None

        def post(self, url: str, data: dict[str, str], headers: dict[str, str]) -> FakeSyncResponse:
            del url, data, headers
            html = """
            <html>
              <a class='result-link' href='https://example.com'>Example</a>
              <td class='result-snippet'>Snippet text</td>
            </html>
            """
            return FakeSyncResponse(html)

    monkeypatch.setattr(websearch_tool.httpx, "Client", FakeSyncClient)
    parsed = websearch_tool._search_duckduckgo_lite("query", 5, 10)
    assert parsed == [{"link": "https://example.com", "title": "Example", "snippet": "Snippet text"}]

    chromedriver = "chromedriver-for-test"
    monkeypatch.setenv("CHROMEDRIVER_PATH", chromedriver)
    monkeypatch.setattr(websearch_tool.os.path, "isfile", lambda path: path == chromedriver)
    service = websearch_tool._get_chromedriver_service()
    assert getattr(service, "path", None) == chromedriver or getattr(service, "service_args", None) is not None

    class FakeTitleElement:
        def __init__(self, title: str, link: str) -> None:
            self.text = title
            self._link = link

        def get_attribute(self, name: str) -> str:
            assert name == "href"
            return self._link

    class FakeEntry:
        def __init__(self, title: str, link: str, snippet: str) -> None:
            self.title = title
            self.link = link
            self.snippet = snippet

        def find_element(self, by: Any, value: str) -> FakeTitleElement:
            del by, value
            return FakeTitleElement(self.title, self.link)

        def find_elements(self, by: Any, value: str) -> list[Any]:
            del by, value
            return [SimpleNamespace(text=self.snippet)]

    class FakeDriver:
        def __init__(self) -> None:
            self.closed = False

        def get(self, url: str) -> None:
            self.url = url

        def find_elements(self, by: Any, value: str) -> list[FakeEntry]:
            del by, value
            return [FakeEntry("Title", "https://example.org", "Snippet")]

        def quit(self) -> None:
            self.closed = True

    fake_driver = FakeDriver()
    monkeypatch.setattr(websearch_tool, "_get_chromedriver_service", lambda: "service")
    monkeypatch.setattr(websearch_tool.webdriver, "Chrome", lambda service, options: fake_driver)
    monkeypatch.setattr(
        websearch_tool, "WebDriverWait", lambda driver, timeout: SimpleNamespace(until=lambda condition: True)
    )
    results = websearch_tool._run_selenium_search("test query", 1, 5)
    assert results[0]["title"] == "Title"
    assert fake_driver.closed is True

    fake_driver = FakeDriver()
    monkeypatch.setattr(websearch_tool.webdriver, "Chrome", lambda service, options: fake_driver)
    monkeypatch.setattr(
        websearch_tool,
        "WebDriverWait",
        lambda driver, timeout: SimpleNamespace(
            until=lambda condition: (_ for _ in ()).throw(TimeoutException("late"))
        ),
    )
    with pytest.raises(RuntimeError, match="timed out"):
        websearch_tool._run_selenium_search("test query", 1, 5)
    assert fake_driver.closed is True

    websearch_tool.register_tools(cast(Any, fake_mcp))

    monkeypatch.setattr(
        websearch_tool,
        "_search_duckduckgo_lite",
        lambda query, num_results, timeout: [{"title": "T", "link": "L", "snippet": "S"}],
    )
    success = await fake_mcp.tools["perform_web_search"]("query", ctx, mcp_session_id="override")
    assert success["results"][0]["title"] == "T"
    assert created_sessions[0] == ("override", "client-2")

    monkeypatch.setattr(websearch_tool, "_search_duckduckgo_lite", lambda query, num_results, timeout: [])
    monkeypatch.setattr(websearch_tool, "_run_selenium_search", lambda query, num_results, timeout: [])
    empty = await fake_mcp.tools["perform_web_search"]("query", ctx)
    assert empty["message"] == "No results found."

    monkeypatch.setattr(
        websearch_tool,
        "_search_duckduckgo_lite",
        lambda query, num_results, timeout: (_ for _ in ()).throw(RuntimeError("search failed")),
    )
    error = await fake_mcp.tools["perform_web_search"]("query", ctx)
    assert "error" in error
