import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def get_model(temperature: float = 0.0):
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    model_name = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")

    return init_chat_model(
        model_name,
        model_provider="google_genai",
        temperature=temperature,
    )

    