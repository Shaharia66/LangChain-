from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from config import get_model

SYSTEM = "You are a concise assistant. Answer in one short sentence."

no_memory_agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt=SYSTEM,
)

memory_agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt=SYSTEM,
    checkpointer=InMemorySaver(),
)


def ask(agent, text, thread_id=None):
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    return result["messages"][-1].text


print("A) NO MEMORY")
ask(no_memory_agent, "My name is Asha.")
print("   ", ask(no_memory_agent, "What is my name?"))

print("\nB) WITH MEMORY, thread 'user-1'")
ask(memory_agent, "My name is Asha.", thread_id="user-1")
print("   ", ask(memory_agent, "What is my name?", thread_id="user-1"))

print("\nC) WITH MEMORY, different thread 'user-2'")
print("   ", ask(memory_agent, "What is my name?", thread_id="user-2"))

print("\nD) INSIDE THREAD 'user-1'")
config = {"configurable": {"thread_id": "user-1"}}
state = memory_agent.get_state(config)
for m in state.values["messages"]:
    print("   ", type(m).__name__, "->", m.text[:60])

