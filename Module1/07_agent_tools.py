from datetime import datetime
from langchain.tools import tool
from langchain.agents import create_agent
from config import get_model


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


@tool
def get_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


SAMPLE_WEATHER = {
    "paris": "18C, cloudy",
    "tokyo": "24C, sunny",
    "dhaka": "31C, humid",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Uses sample data, not live data."""
    return SAMPLE_WEATHER.get(city.strip().lower(), "No data for that city.")


agent = create_agent(
    model=get_model(),
    tools=[multiply, get_time, get_weather],
    system_prompt="You are a helpful assistant. Use tools when they help. Never guess numbers.",
)

question = "What is the weather in Tokyo, and what is 87 times 43?"
result = agent.invoke({"messages": [{"role": "user", "content": question}]})

for message in result["messages"]:
    message.pretty_print()

print("\nFINAL:", result["messages"][-1].text)

