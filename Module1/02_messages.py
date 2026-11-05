from langchain.messages import SystemMessage, HumanMessage
from config import get_model

model = get_model()

# Part A: system + human message
messages = [
    SystemMessage("You are a concise teacher. Answer in at most 2 sentences."),
    HumanMessage("What is a vector database?"),
]
first = model.invoke(messages)
print("Q1:", first.text)

# Part B: follow-up WITHOUT history (model forgets)
no_memory = model.invoke("Give me a real-world analogy for it.")
print("\nWithout history:", no_memory.text)

# Part C: follow-up WITH history (we resend everything)
messages.append(first)
messages.append(HumanMessage("Give me a real-world analogy for it."))
with_memory = model.invoke(messages)
print("\nWith history:", with_memory.text)

# Part D: same thing using plain dictionaries
dict_messages = [
    {"role": "system", "content": "Reply in one short sentence."},
    {"role": "user", "content": "Why is the sky blue?"},
]
print("\nDict format:", model.invoke(dict_messages).text)