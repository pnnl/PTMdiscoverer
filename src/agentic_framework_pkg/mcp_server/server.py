import os

from fastmcp import FastMCP

# Initialize FastMCP instance
mcp = FastMCP(
    name=os.getenv("MCP_SERVER_NAME", "AgenticFrameworkMCP"),
    instructions="This MCP server provides tools for data analysis, general utilities, and scientific workflows.",
    # Add other FastMCP configurations if needed, like default_model for ctx.sample
)

# MCP tool, resource, and prompt definitions will be registered on this 'mcp' instance.
# Example (to be expanded in respective tool files and imported here):
# from.tools.general_tools import register_general_tools
# register_general_tools(mcp)
