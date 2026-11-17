from langchain.agents import create_agent
from langchain.tools import tool

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
)

requests = [
    "Send an email to team@example.com with subject 'Meeting' saying the meeting is at 10am tomorrow.",
    "Send an email to boss@example.com with subject 'Resignation' saying I quit, effective today.",
]

for request in requests:
    print(f"\nUSER: {request}")
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    print("FINAL:", result["messages"][-1].text)