from config import get_model

prompt = "Invent a name for a coffee shop. Reply with only the name."

for temp in (0.0, 1.0):
    model = get_model(temperature=temp)
    print(f"\nTemperature = {temp}")
    for i in range(3):
        print(f"  run {i + 1}:", model.invoke(prompt).text.strip())

