from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from config import get_model


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to one recipient."""
    print(f"   >>> EMAIL SENT (fake) to={to} | subject={subject}")
    return f"Email sent to {to}."


agent = create_agent(
    model=get_model(),
    tools=[send_email],
    system_prompt="You are an email assistant. Use send_email to send emails.",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "hitl-1"}}


def pending_interrupts(result):
    """Return the list of pauses in a result (works for dict or object results)."""
    if isinstance(result, dict):
        return result.get("__interrupt__") or []
    return getattr(result, "interrupts", None) or []


print("=== CALL 1: ask the agent to send an email")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Send an email to team@example.com with subject 'Meeting' saying the meeting is at 10am tomorrow.",
            }
        ]
    },
    config,
)

interrupts = pending_interrupts(result)
if not interrupts:
    print("Not paused. FINAL:", result["messages"][-1].text)
else:
    value = interrupts[0].value
    print("\nPAUSED. No '>>> EMAIL SENT' line appeared above, so nothing was sent.")
    print("RAW INTERRUPT VALUE:", value)
    for action in value.get("action_requests", []):
        print("   tool:", action["name"])
        print("   args:", action["args"])

    state = agent.get_state(config)
    print("   run is waiting (saved):", bool(state.next))

    print("\n=== CALL 2: a human approves")
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    print("FINAL:", result["messages"][-1].text)
    print("   run still waiting:", bool(agent.get_state(config).next))