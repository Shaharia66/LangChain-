import base64
from pathlib import Path
from langchain.messages import HumanMessage
from config import get_model

IMAGE_PATH = Path("sample_invoice.png")
MIME_TYPE = "image/png"

if not IMAGE_PATH.exists():
    raise SystemExit("Run 10_make_sample_image.py first.")

image_b64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Read this invoice. Give the invoice number, customer, and total.",
        },
        {"type": "image", "base64": image_b64, "mime_type": MIME_TYPE},
    ]
)

model = get_model()
response = model.invoke([message])

print(response.text)
print("\nUSAGE:", response.usage_metadata)

