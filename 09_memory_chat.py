import uuid
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from config import get_model

agent = create_agent(
    model=get_model(temperature=0.3),
    tools=[],
    system_prompt="You are a helpful teacher. Keep answers short and clear.",
    checkpointer=InMemorySaver(),
)

thread_id = str(uuid.uuid4())
print("Chat started. Commands: /new = new conversation, exit = quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Bye!")
        break
    if user_input.lower() == "/new":
        thread_id = str(uuid.uuid4())
        print("New conversation started.\n")
        continue
    if not user_input:
        continue

    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}, config
        )
    except Exception as e:
        print("Error:", e)
        continue

    print("AI:", result["messages"][-1].text, "\n")