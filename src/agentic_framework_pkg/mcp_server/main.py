import asyncio
import json
import os
from typing import Any, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastmcp import FastMCP

if os.path.exists(".env"):
    load_dotenv()

from ..core.llm_agnostic_layer import LLMAgnosticClient
from ..logger_config import get_logger  # Use centralized logger
from .server import mcp
from .state_manager import initialize_db
from .tools import (
    blastq_tool,
    csv_rag_tool,
    general_tools,
    ptmdiscoverer_tool,  # PTMdiscoverer integration
    pubchem_tool,
    uniprot_tool,
    websearch_tool,
)

logger = get_logger(__name__)


async def log_request_middleware(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Incoming request: {request.method} {request.url} from {client_host}")
    logger.debug(f"Request Headers: {dict(request.headers)}")

    req_body_bytes = await request.body()

    if req_body_bytes:
        try:
            body_json = json.loads(req_body_bytes.decode("utf-8"))
            logger.info(f"Request body (parsed as JSON): {body_json}")
        except json.JSONDecodeError:
            raw_body = req_body_bytes[:500].decode(errors="ignore")
            logger.info(f"Request body (raw, not JSON): {raw_body}...")
        except UnicodeDecodeError:
            logger.info("Request body (raw, non-UTF-8): %r...", req_body_bytes[:500])
    else:
        logger.info("Request body: Empty")

    async def new_receive():
        return {"type": "http.request", "body": req_body_bytes, "more_body": False}

    new_request = Request(scope=request.scope, receive=new_receive, send=request._send)

    response = await call_next(new_request)
    logger.info(f"Outgoing response: {response.status_code}")
    return response


llm_agnostic_client_instance = LLMAgnosticClient()


def setup_mcp_server(mcp_instance: FastMCP = mcp) -> FastAPI:
    """Sets up the MCP server instance by registering tools."""
    logger.info("Setting up MCP server and registering tools...")
    general_tools.register_tools(mcp_instance)
    csv_rag_tool.register_tools(mcp_instance, llm_agnostic_client_instance)
    uniprot_tool.register_tools(mcp_instance)
    websearch_tool.register_tools(mcp_instance)
    blastq_tool.register_tools(mcp_instance)
    pubchem_tool.register_tools(mcp_instance)
    ptmdiscoverer_tool.register_tools(mcp_instance)
    logger.info("MCP tools registered.")

    actual_fastapi_app = cast(FastAPI, cast(Any, mcp_instance).http_app())
    actual_fastapi_app.middleware("http")(log_request_middleware)
    logger.info("Request logging middleware added.")
    return actual_fastapi_app


async def main_async():
    """Asynchronous main function to initialize DB and start server."""
    logger.info("Initializing database...")
    await initialize_db()

    app = setup_mcp_server()

    server_host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")  # noqa: S104 — intentional, Docker listens on all interfaces
    server_port = int(os.getenv("MCP_SERVER_PORT", "8080"))
    num_workers = int(os.getenv("UVICORN_WORKERS", "1"))

    config = uvicorn.Config(
        app,
        host=server_host,
        port=server_port,
        log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
        workers=num_workers,
    )
    server = uvicorn.Server(config)
    logger.info(f"Starting MCP Server on {server_host}:{server_port} with {num_workers} worker(s)...")
    await server.serve()


def start_server_cli():
    """CLI entry point to start the server."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("MCP Server shutting down...")
    except Exception as e:
        logger.critical(f"MCP Server failed to start or crashed: {e}", exc_info=True)


if __name__ == "__main__":
    start_server_cli()
