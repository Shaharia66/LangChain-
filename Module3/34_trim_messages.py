from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from config import get_model

MAX_MESSAGES = 6


@before_model
def keep_last_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if len(messages) <= MAX_MESSAGES:
        return None  # short enough, change nothing

    kept = messages[-MAX_MESSAGES:]
    while kept and kept[0].type != "human":
        kept = kept[1:]  # never start on an AI or tool message

    print(f"   [trim] {len(messages)} -> {len(kept)} messages")
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}


agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt="You are a teacher. Answer in 2 short sentences.",
    middleware=[keep_last_messages],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "trim-demo"}}

questions = [
    "Hi! My name is Asha and I live in Dhaka.",
    "What is a token?",
    "What is a context window?",
    "What is middleware?",
    "What is a hook?",
    "What is my name?",
]

for turn, question in enumerate(questions, start=1):
    print(f"\nTurn {turn}: {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    print("   ANSWER:", result["messages"][-1].text)
    print("   messages in saved history:", len(result["messages"]))

