import asyncio
import sys
from pathlib import Path

from langchain.mcp import MCPAdapter

from agent_utils import show_tools, allow_only, ask

HERE = Path(__file__).resolve().parent

CONFIG = {
    "mcpServers": {
        "math": {"command": sys.executable, "args": [str(HERE / "math_server.py")]},
        "web": {"url": "http://127.0.0.1:8765/mcp"},
    }
}

ALLOWED = {"math_add", "math_multiply"}


async def main():
    async with MCPAdapter(CONFIG) as adapter:
        offered = await adapter.list_tools()
        show_tools(offered)
        tools = allow_only(offered, ALLOWED)
        await ask(tools, "What is 7 plus 8, then multiplied by 3?")


if __name__ == "__main__":
    asyncio.run(main())