from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from config import get_model


@dataclass
class Context:
    user_id: str


@tool
def save_preference(name: str, value: str, runtime: ToolRuntime[Context]) -> str:
    """Save one food preference for the user, such as name='diet', value='vegetarian'."""
    namespace = ("preferences", runtime.context.user_id)
    item = runtime.store.get(namespace, "prefs")
    prefs = dict(item.value) if item else {}
    prefs[name.strip().lower()] = value.strip()
    runtime.store.put(namespace, "prefs", prefs)
    return f"Saved {name}."


@tool
def get_preferences(runtime: ToolRuntime[Context]) -> str:
    """Get all saved food preferences for the current user."""
    namespace = ("preferences", runtime.context.user_id)
    item = runtime.store.get(namespace, "prefs")
    return str(item.value) if item else "No preferences saved yet."


store = InMemoryStore()

agent = create_agent(
    model=get_model(),
    tools=[save_preference, get_preferences],
    system_prompt=(
        "You are Chef Bot. Save stated food preferences with save_preference. "
        "Use get_preferences before suggesting meals. Keep answers short."
    ),
    checkpointer=InMemorySaver(),
    store=store,
    context_schema=Context,
)


def ask(text, thread_id, user_id):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config,
        context=Context(user_id=user_id),
    )
    return result["messages"][-1].text


print("A) u1, thread t-1:", ask("I am vegetarian and I dislike mushrooms. Remember that.", "t-1", "u1"))
print("B) u1, NEW thread t-2:", ask("What do you know about my food preferences?", "t-2", "u1"))
print("C) u2, thread t-3:", ask("What do you know about my food preferences?", "t-3", "u2"))
print("D) raw store:", store.get(("preferences", "u1"), "prefs").value)

