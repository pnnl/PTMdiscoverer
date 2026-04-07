from __future__ import annotations

import importlib
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request, Response


def _import_real_module(module_name: str) -> Any:
    existing = sys.modules.get(module_name)
    if isinstance(existing, Mock):
        sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


mcp_main = _import_real_module("agentic_framework_pkg.mcp_server.main")
mcp_server_module = _import_real_module("agentic_framework_pkg.mcp_server.server")
sm = _import_real_module("agentic_framework_pkg.mcp_server.state_manager")
vsm = _import_real_module("agentic_framework_pkg.mcp_server.vector_store_manager")
general_tools = _import_real_module("agentic_framework_pkg.mcp_server.tools.general_tools")


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


class FakeCollection:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def get(
        self, ids: list[str] | None = None, where: dict[str, Any] | None = None, include: Any = None
    ) -> dict[str, Any]:
        if ids is not None:
            selected = [(item_id, self.items[item_id]) for item_id in ids if item_id in self.items]
        elif where and where != {}:
            session_id = where.get("session_id")
            selected = [
                (item_id, item)
                for item_id, item in self.items.items()
                if item.get("metadata", {}).get("session_id") == session_id
            ]
        else:
            selected = list(self.items.items())

        return {
            "ids": [item_id for item_id, _ in selected],
            "metadatas": [item.get("metadata") for _, item in selected],
            "documents": [item.get("document") for _, item in selected],
            "embeddings": [item.get("embedding") for _, item in selected],
        }

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        for index, item_id in enumerate(ids):
            self.items[item_id] = {
                "document": documents[index],
                "metadata": metadatas[index],
                "embedding": embeddings[index] if embeddings else None,
            }

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        if ids is not None:
            for item_id in ids:
                self.items.pop(item_id, None)
            return

        if where and where != {}:
            session_id = where.get("session_id")
            to_delete = [
                item_id
                for item_id, item in self.items.items()
                if item.get("metadata", {}).get("session_id") == session_id
            ]
            for item_id in to_delete:
                self.items.pop(item_id, None)
            return

        self.items.clear()

    def query(
        self,
        *,
        query_embeddings: list[list[float]] | None = None,
        n_results: int,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: Any = None,
        query_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        del query_embeddings, where_document, include, query_texts
        selected = self.get(where=where)
        ids = selected["ids"][:n_results]
        documents = selected["documents"][:n_results]
        metadatas = selected["metadatas"][:n_results]
        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [[0.1 for _ in ids]],
        }


async def _immediate_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


@pytest.fixture
def patched_state_manager(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeCollection, FakeCollection]:
    sessions = FakeCollection()
    embeddings = FakeCollection()
    monkeypatch.setattr(sm.asyncio, "to_thread", _immediate_to_thread)
    monkeypatch.setattr(sm, "_sessions_collection", sessions)
    monkeypatch.setattr(sm, "_embeddings_collection", embeddings)
    return sessions, embeddings


@pytest.mark.unit
def test_server_module_exposes_fastmcp_instance() -> None:
    assert mcp_server_module.mcp is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_log_request_middleware_preserves_request_body() -> None:
    captured: dict[str, bytes] = {}

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b'{"x": 1}', "more_body": False}

    async def send(_: MutableMapping[str, Any]) -> None:
        return None

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
    }
    request = Request(scope, receive=receive, send=send)

    async def call_next(new_request: Request) -> Response:
        captured["body"] = await new_request.body()
        return Response("ok", status_code=201)

    response = await mcp_main.log_request_middleware(request, call_next)

    assert response.status_code == 201
    assert captured["body"] == b'{"x": 1}'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_log_request_middleware_handles_non_json_and_missing_client() -> None:
    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"plain-text-body", "more_body": False}

    async def send(_: MutableMapping[str, Any]) -> None:
        return None

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
    }
    request = Request(scope, receive=receive, send=send)

    async def call_next(new_request: Request) -> Response:
        assert await new_request.body() == b"plain-text-body"
        return Response("ok", status_code=200)

    response = await mcp_main.log_request_middleware(request, call_next)
    assert response.status_code == 200


@pytest.mark.unit
def test_setup_mcp_server_registers_all_tool_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    app = FastAPI()

    class FakeServerApp:
        def http_app(self) -> FastAPI:
            return app

    def register_general(instance: Any) -> None:
        del instance
        calls.append("general")

    def register_csv_rag(instance: Any, llm_client: Any) -> None:
        del instance
        if llm_client is mcp_main.llm_agnostic_client_instance:
            calls.append("csv_rag")

    def register_uniprot(instance: Any) -> None:
        del instance
        calls.append("uniprot")

    def register_websearch(instance: Any) -> None:
        del instance
        calls.append("websearch")

    def register_blastq(instance: Any) -> None:
        del instance
        calls.append("blastq")

    def register_pubchem(instance: Any) -> None:
        del instance
        calls.append("pubchem")

    def register_ptm(instance: Any) -> None:
        del instance
        calls.append("ptm")

    monkeypatch.setattr(mcp_main.general_tools, "register_tools", register_general)
    monkeypatch.setattr(mcp_main.csv_rag_tool, "register_tools", register_csv_rag)
    monkeypatch.setattr(mcp_main.uniprot_tool, "register_tools", register_uniprot)
    monkeypatch.setattr(mcp_main.websearch_tool, "register_tools", register_websearch)
    monkeypatch.setattr(mcp_main.blastq_tool, "register_tools", register_blastq)
    monkeypatch.setattr(mcp_main.pubchem_tool, "register_tools", register_pubchem)
    monkeypatch.setattr(mcp_main.ptmdiscoverer_tool, "register_tools", register_ptm)

    result = mcp_main.setup_mcp_server(FakeServerApp())

    assert result is app
    assert calls == ["general", "csv_rag", "uniprot", "websearch", "blastq", "pubchem", "ptm"]
    assert app.user_middleware


@pytest.mark.unit
@pytest.mark.asyncio
async def test_main_async_initializes_db_and_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    app = FastAPI()

    async def fake_initialize_db() -> None:
        captured["db_initialized"] = True

    def fake_setup_mcp_server() -> FastAPI:
        captured["app_created"] = True
        return app

    class FakeConfig:
        def __init__(self, app: FastAPI, host: str, port: int, log_level: str, workers: int) -> None:
            captured.update({"config_app": app, "host": host, "port": port, "log_level": log_level, "workers": workers})

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            captured["config"] = config

        async def serve(self) -> None:
            captured["served"] = True

    monkeypatch.setattr(mcp_main, "initialize_db", fake_initialize_db)
    monkeypatch.setattr(mcp_main, "setup_mcp_server", fake_setup_mcp_server)
    monkeypatch.setattr(mcp_main.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(mcp_main.uvicorn, "Server", FakeServer)
    monkeypatch.setenv("MCP_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MCP_SERVER_PORT", "9001")
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    await mcp_main.main_async()

    assert captured["db_initialized"] is True
    assert captured["app_created"] is True
    assert captured["config_app"] is app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9001
    assert captured["workers"] == 2
    assert captured["log_level"] == "debug"
    assert captured["served"] is True


@pytest.mark.unit
def test_start_server_cli_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(coro: Any) -> None:
        coro.close()
        raise KeyboardInterrupt()

    monkeypatch.setattr(mcp_main.asyncio, "run", fake_run)
    mcp_main.start_server_cli()


@pytest.mark.unit
def test_start_server_cli_logs_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []

    def fake_run(coro: Any) -> None:
        coro.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(mcp_main.asyncio, "run", fake_run)
    monkeypatch.setattr(mcp_main.logger, "critical", lambda msg, exc_info=True: messages.append(msg))
    mcp_main.start_server_cli()
    assert messages and "crashed" in messages[0]


@pytest.mark.unit
def test_get_chroma_client_sync_persistent_and_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    persistent_calls: list[Path] = []
    memory_calls: list[str] = []

    monkeypatch.setattr(sm, "_client", None)
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "db"))
    monkeypatch.setattr(sm.os, "makedirs", lambda path, exist_ok: persistent_calls.append(Path(path)))
    monkeypatch.setattr(sm.chromadb, "PersistentClient", lambda path: {"persistent": path})

    assert sm._get_chroma_client_sync() == {"persistent": str(tmp_path / "db")}
    assert persistent_calls == [tmp_path / "db"]

    monkeypatch.setattr(sm, "_client", None)
    monkeypatch.setenv("CHROMA_DB_PATH", ":memory:")

    def fake_memory_client() -> dict[str, bool]:
        memory_calls.append("memory")
        return {"memory": True}

    monkeypatch.setattr(sm.chromadb, "Client", fake_memory_client)

    assert sm._get_chroma_client_sync() == {"memory": True}
    assert memory_calls == ["memory"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_initialize_db_sets_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = FakeCollection()
    embeddings = FakeCollection()

    class FakeClient:
        def get_or_create_collection(self, name: str) -> FakeCollection:
            return sessions if name == sm.SESSIONS_COLLECTION_NAME else embeddings

    monkeypatch.setattr(sm.asyncio, "to_thread", _immediate_to_thread)
    monkeypatch.setattr(sm, "_get_chroma_client_sync", lambda: FakeClient())

    await sm.initialize_db()

    assert sm._get_sessions_collection() is sessions
    assert sm._get_embeddings_collection() is embeddings


@pytest.mark.unit
def test_get_collection_helpers_raise_when_uninitialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_sessions_collection", None)
    monkeypatch.setattr(sm, "_embeddings_collection", None)
    with pytest.raises(RuntimeError):
        sm._get_sessions_collection()
    with pytest.raises(RuntimeError):
        sm._get_embeddings_collection()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_context_lifecycle(patched_state_manager: tuple[FakeCollection, FakeCollection]) -> None:
    sessions, _ = patched_state_manager

    await sm.create_session_if_not_exists("s1", client_id="c1", initial_context={"hello": "world"})
    context = await sm.get_session_context("s1")
    assert context == {"hello": "world"}

    await sm.update_session_context("s1", {"updated": True})
    updated_context = await sm.get_session_context("s1")
    assert updated_context == {"updated": True}

    metadata = await sm.get_session_metadata("s1")
    assert metadata["client_id"] == "c1"
    assert "created_at" in metadata

    await sm.update_session_metadata("s1", {"extra": "value"})
    assert (await sm.get_session_metadata("s1"))["extra"] == "value"
    assert await sm.get_session_ids() == ["s1"]
    assert (await sm.get_all_sessions())["ids"] == ["s1"]

    await sm.delete_session_metadata("s1")
    assert sessions.items == {}
    await sm.delete_session_metadata("missing")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_session_context_handles_invalid_json(
    patched_state_manager: tuple[FakeCollection, FakeCollection],
) -> None:
    sessions, _ = patched_state_manager
    sessions.items["s1"] = {
        "document": "s1",
        "metadata": {"context_data": "{not-json", "client_id": "c1"},
        "embedding": None,
    }

    assert await sm.get_session_context("s1") == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embeddings_lifecycle_and_delete_session_cascade(
    patched_state_manager: tuple[FakeCollection, FakeCollection],
) -> None:
    sessions, embeddings = patched_state_manager

    await sm.create_session_if_not_exists("s1", client_id="c1")
    await sm.add_embeddings_batch(
        ids=["e1", "e2"],
        documents=["doc1", "doc2"],
        embeddings=[[0.1], [0.2]],
        metadatas=[{"session_id": "s1"}, {"session_id": "other"}],
    )

    assert (await sm.get_embeddings())["ids"] == ["e1", "e2"]
    query = await sm.query_embeddings([[0.5]], where={"session_id": "s1"})
    assert query["ids"] == [["e1"]]

    await sm.delete_embeddings(where={"session_id": "other"})
    assert "e2" not in embeddings.items

    await sm.delete_session("s1")
    assert "s1" not in sessions.items
    assert "e1" not in embeddings.items

    await sm.add_embeddings_batch(
        ids=["e3"],
        documents=["doc3"],
        embeddings=[[0.3]],
        metadatas=[{"session_id": "x"}],
    )
    assert (await sm.get_all_embeddings())["ids"] == ["e3"]
    await sm.clear_embeddings()
    await sm.clear_sessions()
    assert (await sm.get_all_embeddings())["ids"] == []
    assert (await sm.get_all_sessions())["ids"] == []


@pytest.mark.unit
def test_vector_store_manager_add_and_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    class FakeCollectionForVSM:
        def add(self, **kwargs: Any) -> None:
            calls.append(("add", kwargs))

        def query(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("query", kwargs))
            return {"ids": [["doc1"]], "documents": [["hello"]]}

    class FakeClient:
        def __init__(self, path: str) -> None:
            self.path = path
            self.collection = FakeCollectionForVSM()

        def get_or_create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> FakeCollectionForVSM:
            calls.append(("collection", name, metadata))
            return self.collection

    monkeypatch.setattr(vsm.VectorStoreManager, "_instance", None)
    monkeypatch.setattr(vsm.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(vsm.chromadb, "PersistentClient", lambda path: FakeClient(path))

    manager = vsm.VectorStoreManager()
    manager.add_documents("docs", ["hello"], embeddings=[[0.1]], metadatas=[{"a": 1}], ids=["id1"])
    manager.add_documents("docs", ["world"], metadatas=[{"b": 2}], ids=["id2"])
    result = manager.query_collection("docs", query_texts=["find this"])

    assert result["ids"] == [["doc1"]]
    assert any(call[0] == "add" for call in calls)
    assert any(call[0] == "query" for call in calls)


@pytest.mark.unit
def test_vector_store_manager_handles_empty_documents_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class ErrorCollection:
        def add(self, **kwargs: Any) -> None:
            raise RuntimeError("add failed")

        def query(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("query failed")

    class ErrorClient:
        def get_or_create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> ErrorCollection:
            del name, metadata
            return ErrorCollection()

    monkeypatch.setattr(vsm.VectorStoreManager, "_instance", None)
    monkeypatch.setattr(vsm.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(vsm.chromadb, "PersistentClient", lambda path: ErrorClient())

    manager = vsm.VectorStoreManager()
    assert manager.add_documents("docs", []) is None
    with pytest.raises(RuntimeError):
        manager.add_documents("docs", ["hello"])
    with pytest.raises(ValueError):
        manager.query_collection("docs")
    with pytest.raises(RuntimeError):
        manager.query_collection("docs", query_embeddings=[[0.1]])


@pytest.mark.unit
def test_general_tool_helpers_resolve_session_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = FakeCtx(session_id="session-1", request_id="request-1")
    assert general_tools.get_stable_session_id(ctx) == "session-1"

    ctx = FakeCtx(request_id="request-2")
    assert general_tools.get_stable_session_id(ctx) == "request-2"

    monkeypatch.setattr(general_tools.uuid, "uuid4", lambda: "generated-id")
    assert general_tools.get_stable_session_id(FakeCtx()) == "generated-id"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensure_session_initialized_uses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    async def fake_create_session(
        session_id: str, client_id: str | None, initial_context: dict[str, Any] | None = None
    ) -> None:
        del initial_context
        calls.append((session_id, client_id))

    monkeypatch.setattr(general_tools, "create_session_if_not_exists", fake_create_session)

    ctx = FakeCtx(session_id="ctx-session", client_id="client-1")
    result = await general_tools.ensure_session_initialized(ctx, override_session_id="override")

    assert result == "override"
    assert calls == [("override", "client-1")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_general_tool_registration_and_behaviors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCP()
    stored_contexts: list[dict[str, Any]] = []

    async def fake_ensure_session_initialized(ctx: FakeCtx, override_session_id: str | None = None) -> str:
        del ctx
        return override_session_id or "session-1"

    async def fake_get_session_context(session_id: str) -> dict[str, Any]:
        if session_id == "files-session":
            return {"uploaded_csv_files": {"file-1": {"original_filename": "a.csv"}}}
        if session_id == "notes-session":
            return {"user_notes": [{"note": "saved"}]}
        return {"user_notes": []}

    async def fake_update_session_context(session_id: str, new_context: dict[str, Any]) -> None:
        stored_contexts.append({"session_id": session_id, **new_context})

    monkeypatch.setattr(general_tools, "ensure_session_initialized", fake_ensure_session_initialized)
    monkeypatch.setattr(general_tools, "get_session_context", fake_get_session_context)
    monkeypatch.setattr(general_tools, "update_session_context", fake_update_session_context)

    general_tools.register_tools(fake_mcp)
    ctx = FakeCtx()

    current_time = await fake_mcp.tools["get_current_datetime"](ctx, mcp_session_id="time-session")
    calc_success = await fake_mcp.tools["perform_calculation"]("2 + 3", ctx, mcp_session_id="calc-session")
    calc_error = await fake_mcp.tools["perform_calculation"]("__import__('os')", ctx, mcp_session_id="calc-session")
    store_result = await fake_mcp.tools["store_note_in_session"]("remember this", ctx, mcp_session_id="session-1")
    notes = await fake_mcp.tools["retrieve_session_notes"](ctx, mcp_session_id="notes-session")
    no_notes = await fake_mcp.tools["retrieve_session_notes"](ctx, mcp_session_id="empty-session")
    files = await fake_mcp.tools["list_uploaded_files"](ctx, mcp_session_id="files-session")
    no_files = await fake_mcp.tools["list_uploaded_files"](ctx, mcp_session_id="empty-session")

    assert current_time.endswith("+00:00")
    assert calc_success == "5.0"
    assert calc_error.startswith("Error:")
    assert store_result == "Note stored successfully in session."
    assert notes["notes"][0]["note"] == "saved"
    assert no_notes["notes"] == []
    assert files["uploaded_files"] == [{"file_id": "file-1", "original_filename": "a.csv"}]
    assert no_files["message"].startswith("No files have been uploaded")
    assert stored_contexts and stored_contexts[0]["user_notes"][0]["note"] == "remember this"
