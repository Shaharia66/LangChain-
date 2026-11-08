import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter

from agent_utils import show_tools, allow_only, ask

SERVER = Path(__file__).resolve().parent / "math_server.py"
ALLOWED = {"add", "multiply"}


async def main():
    async with MCPAdapter(SERVER) as adapter:
        offered = await adapter.list_tools()
        show_tools(offered)
        tools = allow_only(offered, ALLOWED)
        await ask(tools, "What is (12 + 30) times 5?")


if __name__ == "__main__":
    asyncio.run(main())
    