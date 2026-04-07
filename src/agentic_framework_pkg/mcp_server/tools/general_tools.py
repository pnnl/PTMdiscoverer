import ast
import datetime
import operator
import uuid
from collections.abc import Callable
from typing import Any

from fastmcp import Context, FastMCP

from ...logger_config import get_logger
from ..state_manager import create_session_if_not_exists, get_session_context, update_session_context

logger = get_logger(__name__)


def get_stable_session_id(ctx: Context) -> str:
    """
    Retrieve a stable session ID from the MCP context.
    """
    if hasattr(ctx, "session_id") and ctx.session_id:
        return ctx.session_id
    if ctx.request_id:
        return ctx.request_id

    logger.warning(
        "Could not determine a stable session ID from ctx.request_id or ctx.session_id. Generating a new one."
    )
    # This is NOT ideal for true session persistence if the client doesn't resend this ID.
    # A better approach is for the client to manage and send a session_id if the transport doesn't.
    return str(uuid.uuid4())


async def ensure_session_initialized(ctx: Context, override_session_id: str | None = None) -> str:
    """Helper to ensure the session exists in the database.

    If *override_session_id* is provided (non-empty) it is used as the
    application-level session key.  This is important because each
    MCPToolWrapper call opens a fresh HTTP connection, which creates a
    new MCP protocol session (ctx.session_id).  Passing the stable
    Streamlit/application session ID via the tool's mcp_session_id
    parameter ensures that all calls share the same state store.
    """
    session_id = override_session_id or get_stable_session_id(ctx)
    client_id = ctx.client_id if hasattr(ctx, "client_id") else None
    await create_session_if_not_exists(session_id, client_id)
    return session_id


_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_EXPONENT_LIMIT = 1000  # prevent computational DoS via giant exponents


def _safe_eval(expression: str) -> float:
    """Evaluate a simple arithmetic expression without using eval().

    Supports: +, -, *, /, parentheses, integer and float literals, and
    unary +/-. Division and exponentiation by zero, names, function
    calls, and all other constructs raise ValueError or ZeroDivisionError.

    Parameters
    ----------
    expression:
        The arithmetic expression to evaluate.

    Returns
    -------
    float
        The numeric result.

    Raises
    ------
    ValueError
        If the expression contains unsupported syntax or constructs.
    ZeroDivisionError
        If the expression divides by zero.
    SyntaxError
        If the expression cannot be parsed.
    OverflowError
        If an exponent exceeds the safety limit.
    """

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int | float):
                raise ValueError(f"Non-numeric constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            binary_op_type = type(node.op)
            if binary_op_type not in _BINARY_OPS:
                raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")
            left = _eval(node.left)
            right = _eval(node.right)
            return float(_BINARY_OPS[binary_op_type](left, right))
        if isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in _UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
            return float(_UNARY_OPS[unary_op_type](_eval(node.operand)))
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    if not expression or not expression.strip():
        raise ValueError("Empty expression")
    tree = ast.parse(expression.strip(), mode="eval")
    return _eval(tree.body)


def register_tools(mcp: FastMCP):
    @mcp.tool()
    async def get_current_datetime(ctx: Context, mcp_session_id: str | None = None) -> str:
        """Returns the current date and time in ISO format."""
        await ensure_session_initialized(ctx, override_session_id=mcp_session_id)
        await ctx.info("Fetching current date and time.")
        return datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017

    @mcp.tool()
    async def perform_calculation(expression: str, ctx: Context, mcp_session_id: str | None = None) -> str:
        """
        Evaluates a simple mathematical expression.
        Supports basic arithmetic operations (+, -, *, /), numbers, and parentheses.
        Uses a safe AST-based evaluator — no code execution.
        """
        session_id = await ensure_session_initialized(ctx, override_session_id=mcp_session_id)
        await ctx.info(f"Session {session_id}: Performing calculation for: {expression}")
        try:
            result = _safe_eval(expression)
            await ctx.info(f"Calculation result: {result}")
            return str(result)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
            logger.warning(f"Calculation rejected for expression '{expression}': {e}")
            await ctx.error(f"Error during calculation: {str(e)}")
            return f"Error: {str(e)}"

    @mcp.tool()
    async def store_note_in_session(note_text: str, ctx: Context, mcp_session_id: str | None = None) -> str:
        """Stores a note in the current session's context."""
        session_id = await ensure_session_initialized(ctx, override_session_id=mcp_session_id)
        await ctx.info(f"Session {session_id}: Storing note.")

        current_session_data = await get_session_context(session_id)

        notes_list: list[dict[str, str]] = current_session_data.get("user_notes", [])
        notes_list.append({"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "note": note_text})  # noqa: UP017
        current_session_data["user_notes"] = notes_list

        await update_session_context(session_id, current_session_data)
        return "Note stored successfully in session."

    @mcp.tool()
    async def retrieve_session_notes(ctx: Context, mcp_session_id: str | None = None) -> dict[str, Any]:
        """Retrieves all notes stored in the current session's context."""
        session_id = await ensure_session_initialized(ctx, override_session_id=mcp_session_id)
        await ctx.info(f"Session {session_id}: Retrieving notes.")

        current_session_data = await get_session_context(session_id)
        if "user_notes" in current_session_data:
            return {"notes": current_session_data["user_notes"]}
        return {"notes": [], "message": "No notes found for this session."}

    @mcp.tool()
    async def list_uploaded_files(ctx: Context, mcp_session_id: str | None = None) -> dict[str, Any]:
        """Retrieves a list of all files that have been uploaded and processed in the current session."""
        session_id = await ensure_session_initialized(ctx, override_session_id=mcp_session_id)
        await ctx.info(f"Session {session_id}: Retrieving list of uploaded files.")

        current_session_data = await get_session_context(session_id)
        uploaded_files_info = current_session_data.get(
            "uploaded_csv_files", {}
        )  # "uploaded_csv_files" is the key used in csv_rag_tool.py

        if not uploaded_files_info:
            return {"uploaded_files": [], "message": "No files have been uploaded and processed in this session."}

        files_list = []
        for file_id, info in uploaded_files_info.items():
            files_list.append({"file_id": file_id, "original_filename": info.get("original_filename", "N/A")})

        return {"uploaded_files": files_list}

    logger.info("General MCP tools registered.")
