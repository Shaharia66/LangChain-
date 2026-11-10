from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

from config import get_model

# Invented sample data, not real accounts
USERS = {
    "u1": {"name": "Asha", "plan": "free", "balance": 120},
    "u2": {"name": "Rahim", "plan": "pro", "balance": 900},
}


@dataclass
class Context:
    user_id: str


@tool
def get_my_account(runtime: ToolRuntime[Context]) -> str:
    """Get the current user's name, plan, and balance."""
    user = USERS.get(runtime.context.user_id)
    if user is None:
        return "Unknown user."
    return f"Name: {user['name']}, plan: {user['plan']}, balance: ${user['balance']}"


agent = create_agent(
    model=get_model(),
    tools=[get_my_account],
    system_prompt="You are an account assistant. Use the tool for account facts. Never guess.",
    context_schema=Context,
)

question = {"messages": [{"role": "user", "content": "What plan am I on, and what is my balance?"}]}

for uid in ("u1", "u2"):
    result = agent.invoke(question, context=Context(user_id=uid))
    print(uid, "->", result["messages"][-1].text)
    