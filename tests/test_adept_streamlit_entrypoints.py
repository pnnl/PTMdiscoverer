from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


class FakeSessionState(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _ContextManager:
    def __enter__(self) -> _ContextManager:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        return None


class FakeSidebar(_ContextManager):
    def header(self, *args: Any, **kwargs: Any) -> None:
        return None

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def divider(self) -> None:
        return None

    def expander(self, *args: Any, **kwargs: Any) -> _ContextManager:
        return _ContextManager()

    def caption(self, *args: Any, **kwargs: Any) -> None:
        return None

    def text_input(self, *args: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def write(self, *args: Any, **kwargs: Any) -> None:
        return None

    def file_uploader(self, *args: Any, **kwargs: Any) -> None:
        return None

    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def success(self, *args: Any, **kwargs: Any) -> None:
        return None

    def error(self, *args: Any, **kwargs: Any) -> None:
        return None


class FakeStreamlitModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = FakeSessionState()
        self.sidebar = FakeSidebar()
        self.errors: list[str] = []
        self.markdowns: list[str] = []

    def set_page_config(self, **kwargs: Any) -> None:
        self.page_config = kwargs

    def error(self, message: str) -> None:
        self.errors.append(message)

    def stop(self) -> None:
        raise RuntimeError("streamlit stop")

    def title(self, *args: Any, **kwargs: Any) -> None:
        return None

    def header(self, *args: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.markdowns.append(message)

    def caption(self, *args: Any, **kwargs: Any) -> None:
        return None

    def container(self) -> _ContextManager:
        return _ContextManager()

    def chat_message(self, role: str) -> _ContextManager:
        del role
        return _ContextManager()

    def chat_input(self, *args: Any, **kwargs: Any) -> None:
        return None

    def spinner(self, *args: Any, **kwargs: Any) -> _ContextManager:
        return _ContextManager()

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def divider(self) -> None:
        return None

    def expander(self, *args: Any, **kwargs: Any) -> _ContextManager:
        return _ContextManager()

    def file_uploader(self, *args: Any, **kwargs: Any) -> None:
        return None

    def write(self, *args: Any, **kwargs: Any) -> None:
        return None

    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def success(self, *args: Any, **kwargs: Any) -> None:
        return None

    def text_input(self, *args: Any, **kwargs: Any) -> None:
        return None

    def rerun(self) -> None:
        self.reran = True


class FakeAgent:
    def __init__(self, mcp_session_id: str | None = None) -> None:
        self.mcp_session_id = mcp_session_id
        self.tools = [SimpleNamespace(name="Tool1", description="Does one thing. More detail.")]

    async def arun(
        self, user_input: str, chat_history: list[dict[str, str]] | None = None, callbacks: list[Any] | None = None
    ) -> dict[str, Any]:
        del chat_history, callbacks
        return {"output": f"response:{user_input}"}


def _import_streamlit_app(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, FakeStreamlitModule]:
    fake_st = FakeStreamlitModule()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setitem(
        sys.modules,
        "agentic_framework_pkg.scientific_workflow.langchain_agent",
        SimpleNamespace(ScientificWorkflowAgent=FakeAgent),
    )
    monkeypatch.setitem(sys.modules, "fastmcp", SimpleNamespace(Client=object))
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.agents",
        SimpleNamespace(AgentAction=SimpleNamespace, AgentFinish=SimpleNamespace),
    )
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks.base", SimpleNamespace(BaseCallbackHandler=object))
    sys.modules.pop("agentic_framework_pkg.streamlit_app.app", None)
    module = importlib.import_module("agentic_framework_pkg.streamlit_app.app")
    return module, fake_st


@pytest.mark.unit
def test_start_harness_cli_success_and_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agentic_framework_pkg.streamlit_app import main as streamlit_main

    commands: list[list[str]] = []
    waits: list[bool] = []

    class FakeProcess:
        def wait(self) -> None:
            waits.append(True)

    monkeypatch.setattr(streamlit_main.os.path, "exists", lambda path: path == ".env")
    monkeypatch.setattr(streamlit_main, "load_dotenv", lambda *args, **kwargs: None)

    def fake_popen(cmd: list[str]) -> FakeProcess:
        commands.append(cmd)
        return FakeProcess()

    monkeypatch.setattr(
        streamlit_main.subprocess,
        "Popen",
        fake_popen,
    )
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "9999")
    monkeypatch.setenv("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    streamlit_main.start_harness_cli()
    assert commands and commands[0][0:2] == ["streamlit", "run"]
    assert "9999" in commands[0]
    assert waits == [True]

    monkeypatch.setattr(streamlit_main.subprocess, "Popen", lambda cmd: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(SystemExit, match="1"):
        streamlit_main.start_harness_cli()
    assert "streamlit' command not found" in capsys.readouterr().err

    monkeypatch.setattr(streamlit_main.subprocess, "Popen", lambda cmd: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(SystemExit, match="1"):
        streamlit_main.start_harness_cli()
    assert "Error running Streamlit: boom" in capsys.readouterr().err


@pytest.mark.unit
def test_streamlit_app_import_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    app_module, fake_st = _import_streamlit_app(monkeypatch)

    assert fake_st.page_config["page_title"] == "PTMdiscoverer (with ADEPT-core)"
    assert "mcp_session_id" in fake_st.session_state
    assert fake_st.session_state.agent_instance is not None
    assert app_module._normalize_llm_output("x") == "x"
    assert app_module._normalize_llm_output([{"text": "a"}, "b"]) == "ab"
    assert app_module._normalize_llm_output(123) == "123"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streamlit_app_callbacks_and_agent_response(monkeypatch: pytest.MonkeyPatch) -> None:
    app_module, fake_st = _import_streamlit_app(monkeypatch)

    handler = app_module.StreamlitThoughtsCallbackHandler()
    action = SimpleNamespace(tool="tool-name", tool_input={"x": 1}, log="Thought line\nAction: skip")
    finish = SimpleNamespace(return_values={"output": [{"text": "done"}]})

    handler.on_llm_start({}, ["prompt"])
    handler.on_agent_action(action)
    handler.on_tool_end("result text", run_id=object(), name="ToolA")
    handler.on_agent_finish(finish)
    handler.on_chain_error(RuntimeError("chain fail"))
    handler.on_tool_error(RuntimeError("tool fail"))

    thoughts = fake_st.session_state.agent_thoughts
    assert any("LLM thinking" in item for item in thoughts)
    assert any("Tool Output (ToolA)" in item for item in thoughts)
    assert any("Agent Finished" in item for item in thoughts)

    response = await app_module.get_agent_response("hello", FakeAgent(), [], [handler])
    assert response == "response:hello"

    class BlankAgent:
        async def arun(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {"output": ""}

    blank = await app_module.get_agent_response("hello", BlankAgent(), [], [handler])
    assert blank.startswith("The agent completed its tool calls")

    no_agent = await app_module.get_agent_response("hello", None, [], [handler])
    assert no_agent == "Agent not initialized. Please check server logs."

    class FailingAgent:
        async def arun(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("bad agent")

    failed = await app_module.get_agent_response("hello", FailingAgent(), [], [handler])
    assert failed.startswith("An error occurred while processing your request")
