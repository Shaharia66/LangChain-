import time
from config import get_model

model = get_model()

# Streaming: print tokens as they arrive
print("STREAMING:")
start = time.time()
for chunk in model.stream("Give me 3 short tips for learning Python."):
    print(chunk.text, end="", flush=True)
print(f"\n[done in {time.time() - start:.1f}s]\n")

# Batch: several prompts in one call
print("BATCH:")
questions = [
    "What is the capital of France? One word.",
    "What is 12 * 12? Just the number.",
    "Name one programming language. One word.",
]
answers = model.batch(questions, config={"max_concurrency": 2})
for q, a in zip(questions, answers):
    print(f"- {q} -> {a.text.strip()}")
    