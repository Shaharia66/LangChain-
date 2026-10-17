from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (600, 420), "white")
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.load_default(size=28)
except TypeError:
    font = ImageFont.load_default()

lines = ["MY FRIDGE LIST", "- 6 eggs", "- 3 tomatoes", "- 1 onion", "- cheese", "- rice (in cupboard)"]

y = 25
for line in lines:
    draw.text((40, y), line, fill="black", font=font)
    y += 60

draw.rectangle((10, 10, 589, 409), outline="black", width=3)
img.save("fridge_list.png")
print("Saved fridge_list.png")
