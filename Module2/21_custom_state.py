from langchain.agents import create_agent, AgentState
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from config import get_model


class ChefState(AgentState):
    allergies: str


@tool
def save_allergies(allergies: str, runtime: ToolRuntime[None, ChefState]) -> Command:
    """Save the user's allergies for this conversation. allergies is a comma-separated list."""
    cleaned = ", ".join(a.strip().lower() for a in allergies.split(",") if a.strip())
    return Command(
        update={
            "allergies": cleaned,
            "messages": [
                ToolMessage(f"Saved allergies: {cleaned}", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool
def get_allergies(runtime: ToolRuntime[None, ChefState]) -> str:
    """Return the allergies saved for this conversation."""
    return runtime.state.get("allergies", "none recorded")


agent = create_agent(
    model=get_model(),
    tools=[save_allergies, get_allergies],
    system_prompt=(
        "You are Chef Bot. When the user states allergies, call save_allergies. "
        "When asked about allergies, call get_allergies. Keep answers short."
    ),
    state_schema=ChefState,
    checkpointer=InMemorySaver(),
)


def ask(text, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    return result["messages"][-1].text


print("T1 turn 1:", ask("I'm allergic to eggs and peanuts. Please remember that.", "t-1"))
print("T1 turn 2:", ask("What allergies do I have?", "t-1"))
print("T2 (new thread):", ask("What allergies do I have?", "t-2"))

state = agent.get_state({"configurable": {"thread_id": "t-1"}})
print("RAW STATE FIELD:", state.values.get("allergies"))


