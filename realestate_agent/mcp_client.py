"""MCP client: starts the server as a subprocess and talks to it over stdio."""
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .config import ROOT


class ListingsMCPClient:
    """Async context manager. Enter and exit it from the same asyncio task."""

    async def __aenter__(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "realestate_agent.mcp_server"],
            cwd=str(ROOT),
            # Pass the parent's import path so the server finds the same packages, including
            # on hosts that vendor dependencies outside site-packages.
            env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
        )
        self._transport = stdio_client(params)
        read, write = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc):
        await self._session.__aexit__(*exc)
        await self._transport.__aexit__(*exc)

    async def list_tools(self) -> list[dict]:
        """Return tools in Claude API format, sorted so the tool list stays stable."""
        result = await self._session.list_tools()
        tools = [
            {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
            for t in result.tools
        ]
        return sorted(tools, key=lambda t: t["name"])

    async def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Return (text, is_error) from a real MCP tool invocation."""
        result = await self._session.call_tool(name, arguments)
        text = "\n".join(c.text for c in result.content if c.type == "text")
        return text, bool(result.isError)
