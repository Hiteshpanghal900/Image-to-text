import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

words = ["hitesh", "pytorch", "openai", "quizmaster", "ocr", "image", "text", "flask", "redis", "celery"]
num_images = 100
img_width, img_height = 128, 32
fonts = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
]

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")
images_dir = os.path.join(data_dir, "images")
labels_file = os.path.join(data_dir, "labels.txt")
os.makedirs(images_dir, exist_ok=True)


# Function to add noise to an image
def add_noise(img):
    np_img = np.array(img)
    noise = np.random.randint(0, 40, (img.size[1], img.size[0])).astype('uint8')
    np_img = np.clip(np_img + noise, 0, 255)
    return Image.fromarray(np_img)

with open(labels_file, "w") as f:
    for i in range(num_images):
        text = random.choice(words)

        img = Image.new("L", (img_height, img_width), color=255)
        draw = ImageDraw.Draw(img)

        try:
            font_path = random.choice(fonts)
            font_size = random.randint(16, 24)
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        x = random.randint(5, 20)
        y = random.randint(0, 10)
        draw.text((x,y), text, font=font, fill=random.randint(0,40))

        angle = random.randint(-5, 5)
        img = img.rotate(angle, expand=0, fillcolor=255)

        # if random.random() > 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.0)))

        # img = add_noise(img)

        filename = f"img_{i}.png"
        img.save(os.path.join(images_dir, filename))

        f.write(f"{filename}\t{text}\n")