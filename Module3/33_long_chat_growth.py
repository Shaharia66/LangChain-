from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config import get_model

agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt="You are a teacher. Answer in 2 short sentences.",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "growth-demo"}}

questions = [
    "What is a token?",
    "What is a context window?",
    "What is a checkpointer?",
    "What is middleware?",
    "What is a store?",
    "What is a hook?",
]

for turn, question in enumerate(questions, start=1):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    messages = result["messages"]
    usage = messages[-1].usage_metadata or {}
    print(f"Turn {turn}: messages in history = {len(messages)} | input_tokens = {usage.get('input_tokens', '?')}")

