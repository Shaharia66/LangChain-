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

REQUEST = "Send an email to team@example.com with subject 'Meeting' saying the meeting is at 10am tomorrow."


def pending_interrupts(result):
    if isinstance(result, dict):
        return result.get("__interrupt__") or []
    return getattr(result, "interrupts", None) or []


# ---- Three ways a human can decide ----
def approve(action):
    return {"type": "approve"}


def edit(action):
    new_args = {**action["args"], "to": "manager@example.com"}  # copy all, change one
    return {"type": "edit", "edited_action": {"name": action["name"], "args": new_args}}


def reject(action):
    return {"type": "reject", "message": "Do not send this email. Tell the user it was not sent."}


def run_scenario(label, thread_id, make_decision):
    print(f"\n==================== {label}")
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke({"messages": [{"role": "user", "content": REQUEST}]}, config)
    interrupts = pending_interrupts(result)
    if not interrupts:
        print("Not paused. FINAL:", result["messages"][-1].text)
        return

    action = interrupts[0].value["action_requests"][0]
    print("PAUSED. The agent wants:", action["name"], action["args"])

    decision = make_decision(action)
    print("HUMAN DECISION:", decision["type"])

    result = agent.invoke(Command(resume={"decisions": [decision]}), config)
    if pending_interrupts(result):
        print("The agent paused AGAIN (it proposed another action).")
    else:
        print("FINAL:", result["messages"][-1].text)


run_scenario("A) APPROVE", "decide-a", approve)
run_scenario("B) EDIT (change the recipient)", "decide-b", edit)
run_scenario("C) REJECT", "decide-c", reject)
