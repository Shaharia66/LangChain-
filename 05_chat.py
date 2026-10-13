from langchain.messages import SystemMessage, HumanMessage
from config import get_model

model = get_model(temperature=0.3)

history = [SystemMessage("You are a helpful teacher. Keep answers short and clear.")]

print("Chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Bye!")
        break
    if not user_input:
        continue

    history.append(HumanMessage(user_input))

    try:
        reply = model.invoke(history)
    except Exception as e:
        print("Error:", e)
        history.pop()
        continue

    print("AI:", reply.text, "\n")
    history.append(reply)