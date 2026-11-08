import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter

SERVER = Path(__file__).resolve().parent / "math_server.py"


async def main():
    async with MCPAdapter(SERVER) as adapter:
        tools = await adapter.list_tools()
        print("TOOLS:", [t.name for t in tools])
        add = next(t for t in tools if t.name == "add")
        print("add(2, 3) ->", await add.ainvoke({"a": 2, "b": 3}))


if __name__ == "__main__":
    asyncio.run(main())

    