from config import get_model

model = get_model()

response = model.invoke("Explain what a large language model is in two sentences.")

print("TYPE :", type(response).__name__)
print("TEXT :", response.text)
print("USAGE:", response.usage_metadata)
print("META :", response.response_metadata)
