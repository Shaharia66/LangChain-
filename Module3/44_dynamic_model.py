import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

BASIC_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
STRONG_NAME = os.getenv("STRONG_MODEL_NAME", BASIC_NAME)

basic_model = init_chat_model(BASIC_NAME, model_provider="google_genai")
strong_model = init_chat_model(STRONG_NAME, model_provider="google_genai")

LONG_QUESTION_CHARS = 80


@dataclass
class Context:
    user_name: str
    plan: str = "free"  # "free" or "premium"


def last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.type == "human":
            return m.text
    return ""


@wrap_model_call
def pick_model(request: ModelRequest, handler):
    question = last_user_text(request.messages)
    plan = request.runtime.context.plan

    if plan == "premium":
        label, reason, model = "strong", "premium user", strong_model
    elif len(question) > LONG_QUESTION_CHARS:
        label, reason, model = "strong", "long question", strong_model
    else:
        label, reason, model = "basic", "short question", basic_model

    print(f"   [route] {label} model ({reason})")
    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,  # the default; the middleware may replace it per call
    tools=[],
    system_prompt="You are a helpful teacher. Answer in 2 short sentences.",
    middleware=[pick_model],
    context_schema=Context,
)

tests = [
    ("Hi!", Context("Asha", "free")),
    (
        "Explain step by step how a checkpointer, a store, and a thread_id work together in an agent.",
        Context("Asha", "free"),
    ),
    ("Hi!", Context("Rahim", "premium")),
]

for question, ctx in tests:
    print(f"\nUSER ({ctx.plan}): {question[:60]}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, context=ctx)
    meta = result["messages"][-1].response_metadata
    print("   answered by:", meta.get("model_name", "?"))
    print("   ANSWER:", result["messages"][-1].text)

    