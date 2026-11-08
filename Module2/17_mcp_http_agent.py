import asyncio

from langchain.mcp import MCPAdapter

from agent_utils import show_tools, allow_only, ask

URL = "http://127.0.0.1:8765/mcp"
ALLOWED = {"add"}


async def main():
    async with MCPAdapter(URL) as adapter:
        offered = await adapter.list_tools()
        show_tools(offered)
        tools = allow_only(offered, ALLOWED)
        await ask(tools, "What is 10 plus 5?")


if __name__ == "__main__":
    asyncio.run(main())

