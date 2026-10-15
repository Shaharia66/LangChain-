from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (700, 380), "white")
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.load_default(size=28)
except TypeError:
    font = ImageFont.load_default()

lines = [
    "INVOICE #1042",
    "Customer: Sample Co. Ltd",
    "Notebook x 10 ........ $50",
    "Pen x 20 ............. $20",
    "TOTAL: $70",
]

y = 30
for line in lines:
    draw.text((40, y), line, fill="black", font=font)
    y += 60

draw.rectangle((10, 10, 689, 369), outline="black", width=3)
img.save("sample_invoice.png")
print("Saved sample_invoice.png")