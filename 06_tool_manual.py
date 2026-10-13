from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage
from config import get_model


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


model = get_model()
model_with_tools = model.bind_tools([multiply])

messages = [HumanMessage("What is 1234 times 5678?")]

ai_msg = model_with_tools.invoke(messages)
print("Tool calls requested:", ai_msg.tool_calls)

messages.append(ai_msg)
for call in ai_msg.tool_calls:
    result = multiply.invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final = model_with_tools.invoke(messages)
print("Final answer:", final.text)