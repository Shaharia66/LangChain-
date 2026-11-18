from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain.tools import tool

from config import get_model


@dataclass
class Context:
    user_name: str
    role: str


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (sample data)."""
    return {"paris": "18C, cloudy", "tokyo": "24C, sunny"}.get(city.strip().lower(), "No data.")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@wrap_model_call
def inspect_request(request: ModelRequest, handler):
    tool_names = [getattr(t, "name", str(t)) for t in request.tools]
    ctx = request.runtime.context
    model_name = getattr(request.model, "model", type(request.model).__name__)
    print(
        f"   [request] model={model_name} | tools={tool_names} | "
        f"messages={len(request.messages)} | user={ctx.user_name} | role={ctx.role}"
    )
    return handler(request)  # unchanged: we only look


agent = create_agent(
    model=get_model(),
    tools=[get_weather, multiply],
    system_prompt="You are a helpful assistant. Use tools when they help. Keep answers short.",
    middleware=[inspect_request],
    context_schema=Context,
)

for question, ctx in [
    ("What is the weather in Tokyo?", Context("Asha", "student")),
    ("Say hello in one short sentence.", Context("Rahim", "admin")),
]:
    print(f"\nUSER ({ctx.role}): {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, context=ctx)
    print("FINAL:", result["messages"][-1].text)

    