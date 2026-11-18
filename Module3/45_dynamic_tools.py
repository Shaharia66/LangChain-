from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain.tools import tool

from config import get_model


@dataclass
class Context:
    user_name: str
    role: str  # "student" or "admin"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (sample data)."""
    return {"paris": "18C, cloudy", "tokyo": "24C, sunny"}.get(city.strip().lower(), "No data.")


@tool
def get_staff_report() -> str:
    """Get the confidential staff report (admin only, sample data)."""
    return "Staff report (sample): 3 open requests, 1 late payment."


ADMIN_ONLY = {"get_staff_report"}


@wrap_model_call
def filter_tools(request: ModelRequest, handler):
    role = request.runtime.context.role
    if role == "admin":
        allowed = list(request.tools)
    else:
        allowed = [t for t in request.tools if getattr(t, "name", "") not in ADMIN_ONLY]
    print(f"   [tools] role={role} -> {[getattr(t, 'name', '?') for t in allowed]}")
    return handler(request.override(tools=allowed))


agent = create_agent(
    model=get_model(),
    tools=[get_weather, get_staff_report],  # the FULL list
    system_prompt="You are an office assistant. Use tools when they help. Keep answers short.",
    middleware=[filter_tools],
    context_schema=Context,
)

for ctx in (Context("Asha", "student"), Context("Rahim", "admin")):
    print(f"\nUSER ({ctx.role}): Show me the staff report.")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Show me the staff report."}]},
        context=ctx,
    )
    print("FINAL:", result["messages"][-1].text)

    