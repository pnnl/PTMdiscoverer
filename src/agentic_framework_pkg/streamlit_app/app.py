import asyncio
import json  # For displaying dict results
import os
import re
import uuid
from typing import Any
from uuid import UUID

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Adjust the import path based on your project structure if this file is moved
from agentic_framework_pkg.logger_config import get_logger  # noqa: E402
from agentic_framework_pkg.scientific_workflow.langchain_agent import (  # noqa: E402
    ScientificWorkflowAgent,
)

try:
    from fastmcp import Client as MCPClient
    from langchain_core.agents import AgentAction, AgentFinish

    # Langchain callback imports
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    st.error("FastMCP client library not found. Please ensure 'fastmcp' is installed.")
    st.stop()

logger = get_logger(__name__)

# --- Configuration ---
st.set_page_config(page_title="PTMdiscoverer (with ADEPT-core)", layout="wide")

# MCP Server URL - read from environment with fallback to localhost
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
logger.info(f"Streamlit App: Effective MCP_SERVER_URL is {MCP_SERVER_URL}")

# Determine shared upload directory.
SHARED_UPLOAD_DIR = os.getenv("SHARED_UPLOAD_DIR", "./data/uploaded_files")
logger.info(f"Streamlit App: Effective SHARED_UPLOAD_DIR is {SHARED_UPLOAD_DIR}")
os.makedirs(SHARED_UPLOAD_DIR, exist_ok=True)

# --- Session State Initialization ---
if "mcp_session_id" not in st.session_state:
    st.session_state.mcp_session_id = str(uuid.uuid4())
    logger.info(f"New Streamlit session started. MCP Session ID: {st.session_state.mcp_session_id}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores {"role": "user/assistant", "content": "..."}

if "agent_thoughts" not in st.session_state:
    st.session_state.agent_thoughts = []

if "agent_instance" not in st.session_state:
    try:
        st.session_state.agent_instance = ScientificWorkflowAgent(mcp_session_id=st.session_state.mcp_session_id)
        logger.info(f"ScientificWorkflowAgent initialized for session {st.session_state.mcp_session_id}")
    except ValueError as e:
        st.error(f"Failed to initialize agent: {e}")
        logger.error(
            f"Agent initialization error for session {st.session_state.mcp_session_id}: {e}",
            exc_info=True,
        )
        st.session_state.agent_instance = None
    except Exception as e:
        st.error(f"An unexpected error occurred during agent initialization: {e}")
        logger.error(
            f"Unexpected agent initialization error for session {st.session_state.mcp_session_id}: {e}",
            exc_info=True,
        )
        st.session_state.agent_instance = None


def _normalize_llm_output(output: Any) -> str:
    """Normalize agent output that may be a plain string or a list of structured
    content blocks (e.g. [{'type': 'text', 'text': '...'}]) returned by newer
    LLM providers."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(output)


# --- Custom Langchain Callback Handler for Streamlit ---
class StreamlitThoughtsCallbackHandler(BaseCallbackHandler):
    """A callback handler that writes agent thoughts to Streamlit session state."""

    def _ensure_agent_thoughts_initialized(self):
        if "agent_thoughts" not in st.session_state:
            st.session_state.agent_thoughts = []

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._ensure_agent_thoughts_initialized()
        st.session_state.agent_thoughts.append("🧠 LLM thinking...")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        thought = f"🛠️ Action: {action.tool}\n"
        thought += f"   Input: {action.tool_input}\n"
        # thought += f"   Log: {action.log.strip()}" # log often contains the thought process
        # Extracting the thought process if it's in action.log
        if action.log:
            log_lines = action.log.split("\n")
            thought_lines = [
                line for line in log_lines if not line.startswith("Invoking") and not line.startswith("Action:")
            ]
            if thought_lines:
                self._ensure_agent_thoughts_initialized()
                # Evaluate the expression part separately to avoid f-string parsing issues
                thought_content = "\n".join(thought_lines).strip()
                st.session_state.agent_thoughts.append(f"""🤔 Thought:\n```\n{thought_content}\n```""")
        self._ensure_agent_thoughts_initialized()
        st.session_state.agent_thoughts.append(thought)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._ensure_agent_thoughts_initialized()
        # Evaluate the expression part separately
        tool_name = str(kwargs.get("name", "tool"))
        tool_output_content = str(output)[:500]  # Truncate long outputs
        st.session_state.agent_thoughts.append(
            f"""✅ Tool Output ({tool_name}):\n```\n{tool_output_content}\n```\n---"""
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self._ensure_agent_thoughts_initialized()
        # Evaluate the expression part separately
        final_answer_content = _normalize_llm_output(finish.return_values.get("output", ""))
        st.session_state.agent_thoughts.append(
            f"""🏁 Agent Finished.\nFinal Answer:\n```\n{final_answer_content}\n```"""
        )

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self._ensure_agent_thoughts_initialized()
        st.session_state.agent_thoughts.append(f"❌ Error: {error}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._ensure_agent_thoughts_initialized()
        st.session_state.agent_thoughts.append(f"❌ Tool Error: {error}")


# --- UI Rendering ---
st.title("🧬 PTMdiscoverer (with ADEPT-core)")
st.markdown(
    "**PTMdiscoverer** leverages generative AI to accelerate the discovery of "
    "post-translational modifications (PTMs) in proteomics experiments. "
    "Powered by [ADEPT-core](https://github.com/pnnl/adept-agentic-framework-core), it combines PTM annotation tools "
    "with web search, sequence analysis (BLAST), chemical databases (PubChem, UniProt), "
    "and document RAG — all orchestrated by an LLM agent."
)
st.caption(f"Session: `{st.session_state.mcp_session_id}` · MCP Server: {MCP_SERVER_URL}")

with st.container():
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
user_query = st.chat_input("Ask the agent about PTMs, proteins, or your data…")


async def get_agent_response(
    query: str,
    agent: ScientificWorkflowAgent,
    history: list[dict[str, str]],
    callbacks: list[BaseCallbackHandler],
) -> str:
    """Helper async function to run the agent."""
    if agent is None:
        return "Agent not initialized. Please check server logs."
    try:
        # Assuming agent.arun is modified to accept and use callbacks
        response_dict = await agent.arun(user_input=query, chat_history=history, callbacks=callbacks)
        normalized_output = _normalize_llm_output(response_dict.get("output", ""))
        if not normalized_output.strip():
            logger.warning(
                "Agent finished without a final written answer for session %s. Response keys: %s",
                st.session_state.mcp_session_id,
                sorted(response_dict.keys()),
            )
            return (
                "The agent completed its tool calls but did not produce a final written answer. "
                "This usually means the model ended the chain after tool use without generating a summary. "
                "Review the agent thoughts panel for tool outputs and try a narrower follow-up prompt."
            )
        return normalized_output
    except Exception as e:
        logger.error(
            f"Error during agent execution for session {st.session_state.mcp_session_id}: {e}",
            exc_info=True,
        )
        return f"An error occurred while processing your request: {e}"


if user_query and st.session_state.agent_instance:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.agent_thoughts = []
    streamlit_callback = StreamlitThoughtsCallbackHandler()

    # Get agent response
    with st.spinner("Agent is working… this may take several minutes for multi-run inference."):
        try:
            agent_response_content = asyncio.run(
                get_agent_response(
                    user_query,
                    st.session_state.agent_instance,
                    st.session_state.chat_history[:-1],
                    [streamlit_callback],
                )
            )
        except RuntimeError as e:
            if "cannot be called when another loop is running" in str(e):
                logger.warning("Asyncio loop conflict detected.")
                agent_response_content = "Error: Could not run asynchronous agent due to event loop conflict."
            else:
                raise e

    # Add agent response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": agent_response_content})
    with st.chat_message("assistant"):
        st.markdown(agent_response_content)

elif user_query and not st.session_state.agent_instance:
    st.error("Agent is not available. Please check the application logs or try refreshing.")


# --- Sidebar ---
with st.sidebar:
    st.header("PTMdiscoverer")

    # ---- 1. Reset Chat ----
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent_thoughts = []
        logger.info(f"Chat history cleared for session {st.session_state.mcp_session_id}")
        st.rerun()

    st.divider()

    # ---- 2. Upload + Process Files for RAG ----
    with st.expander("📄 Upload & Process for RAG", expanded=False):
        supported_types = [
            "csv",
            "xlsx",
            "docx",
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "gif",
            "webp",
            "txt",
            "tex",
        ]
        uploaded_csv_file = st.file_uploader(
            f"Upload a document ({', '.join(supported_types)})",
            type=supported_types,
            key="doc_uploader_main_app",
        )

        if uploaded_csv_file is not None:
            st.write(f"**{uploaded_csv_file.name}** ({uploaded_csv_file.size:,} bytes)")
            if st.button("Process for RAG", key="process_csv_button_main_app", use_container_width=True):
                original_filename = uploaded_csv_file.name
                safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
                sanitized_base_filename = "".join(
                    c if c in safe_chars else "_" for c in os.path.splitext(original_filename)[0]
                )
                file_extension = os.path.splitext(original_filename)[1]
                unique_filename_part = str(uuid.uuid4())[:8]
                final_sanitized_filename = f"{sanitized_base_filename}_{st.session_state.mcp_session_id[:8]}_{unique_filename_part}{file_extension}"

                secure_shared_file_path = os.path.join(SHARED_UPLOAD_DIR, final_sanitized_filename)

                try:
                    with open(secure_shared_file_path, "wb") as f:
                        f.write(uploaded_csv_file.getvalue())
                    st.info(f"Saved: {secure_shared_file_path}")

                    async def register_csv_async_rag():
                        try:
                            async with MCPClient(MCP_SERVER_URL) as client:
                                registration_params = {
                                    "file_path_on_server": secure_shared_file_path,
                                    "original_filename": original_filename,
                                    "mcp_session_id": st.session_state.mcp_session_id,
                                }
                                logger.info(f"Calling 'process_uploaded_file' with params: {registration_params}")
                                with st.spinner("Processing document on server…"):
                                    result = await client.call_tool("process_uploaded_file", registration_params)

                                # Normalise FastMCP 2.x (List[ContentPart]) and
                                # FastMCP 3.x (CallToolResult with .content) responses.
                                content_list = (
                                    result.content
                                    if hasattr(result, "content")
                                    else (result if isinstance(result, list) else [])
                                )
                                if content_list:
                                    first_content_item = content_list[0]
                                    if hasattr(first_content_item, "text") and isinstance(first_content_item.text, str):
                                        try:
                                            parsed_data = json.loads(first_content_item.text)
                                            if isinstance(parsed_data, dict) and "file_id" in parsed_data:
                                                st.session_state.current_rag_file_id = parsed_data["file_id"]
                                                st.success(f"Registered with file_id: `{parsed_data['file_id']}`")
                                            elif isinstance(parsed_data, dict) and "error" in parsed_data:
                                                st.error(f"Registration failed: {parsed_data['error']}")
                                            else:
                                                st.warning(f"Unexpected response: {parsed_data}")
                                        except json.JSONDecodeError:
                                            st.error(f"Failed to parse response: {first_content_item.text}")
                                    else:
                                        st.error(f"Unexpected response format: {result}")
                                else:
                                    st.error(f"Empty or unexpected response: {result}")
                        except Exception as e:
                            st.error(f"Error processing file: {e}")
                            logger.error("Error registering document via MCP", exc_info=True)

                    asyncio.run(register_csv_async_rag())

                except Exception as e:
                    st.error(f"Error saving file: {e}")
                    logger.error("Error saving uploaded file", exc_info=True)

        st.caption("Once processed, ask the agent to query the document using its file_id.")
        current_file_id = st.session_state.get("current_rag_file_id", None)
        if current_file_id:
            st.text_input("Current RAG File ID", value=current_file_id, disabled=True)

    st.divider()

    # ---- 3. List Available Tools ----
    with st.expander("🛠️ Available Tools", expanded=False):
        if st.session_state.agent_instance and hasattr(st.session_state.agent_instance, "tools"):
            for tool in st.session_state.agent_instance.tools:
                name = getattr(tool, "name", str(tool))
                desc = getattr(tool, "description", "")
                # Show first sentence of description; split only on period followed
                # by whitespace or end-of-string so "e.g." / "i.e." are preserved.
                _m = re.search(r"\.(?:\s|$)", desc)
                short_desc = desc[: _m.end()].rstrip() if _m else desc
                st.markdown(f"**{name}**  \n{short_desc}")
        else:
            st.caption("Agent not initialized — tools unavailable.")

    st.divider()
    st.caption(f"Session `{st.session_state.mcp_session_id}`")

logger.debug(f"Streamlit app script execution finished for session {st.session_state.mcp_session_id}.")
