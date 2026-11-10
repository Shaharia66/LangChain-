from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from config import get_model


@dataclass
class Context:
    user_name: str
    level: str  # "beginner" or "expert"


@dynamic_prompt
def personalise(request: ModelRequest) -> str:
    ctx = request.runtime.context
    base = f"You are a helpful teacher. Address the user as {ctx.user_name}. Keep answers to 2 sentences."
    if ctx.level == "beginner":
        return base + " Use simple words and a real-world analogy."
    return base + " Be precise and technical."


agent = create_agent(
    model=get_model(),
    tools=[],
    middleware=[personalise],
    context_schema=Context,
)

question = {"messages": [{"role": "user", "content": "What is a checkpointer?"}]}

for ctx in (Context("Asha", "beginner"), Context("Rahim", "expert")):
    result = agent.invoke(question, context=ctx)
    print(f"\n{ctx.user_name} ({ctx.level}):", result["messages"][-1].text)

    