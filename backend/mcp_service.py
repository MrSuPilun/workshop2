"""Small, optional stdio MCP client used for connection validation/discovery."""
from __future__ import annotations

from typing import Any


async def discover_tools(command: str, args: list[str]) -> list[dict[str, Any]]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("Install the optional 'mcp' package to use MCP servers") from exc

    parameters = StdioServerParameters(command=command, args=args)
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {"name": tool.name, "description": tool.description or "", "input_schema": tool.inputSchema}
                for tool in result.tools
            ]


async def call_tool(command: str, args: list[str], tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call one tool using a short-lived stdio connection.

    This keeps the MVP stateless. A production version would pool sessions so
    servers with expensive startup are not relaunched for every call.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=command, args=args)
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            if hasattr(result, "model_dump"):
                return result.model_dump(mode="json")
            return str(result)
