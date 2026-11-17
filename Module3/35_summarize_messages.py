from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from config import get_model

model = get_model()

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a teacher. Answer in 2 short sentences.",
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("messages", 6),
            keep=("messages", 2),
        )
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "summary-demo"}}

questions = [
    "Hi! My name is Asha and I live in Dhaka.",
    "What is a token?",
    "What is a context window?",
    "What is middleware?",
    "What is a hook?",
    "What is my name, and where do I live?",
]

for turn, question in enumerate(questions, start=1):
    print(f"\nTurn {turn}: {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    messages = result["messages"]
    print("   ANSWER:", messages[-1].text)
    print("   messages in saved history:", len(messages))

print("\n--- FIRST MESSAGE IN THE SAVED HISTORY ---")
print(messages[0].type, "->", messages[0].text[:400])


