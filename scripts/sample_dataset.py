from PIL import Image
from zipfile import ZipFile
import os
import random

with ZipFile("sample.zip") as zf:
    for file in random.sample(zf.namelist(), 100):
        if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
            with zf.open(file) as f:
                image = Image.open(f)
                image = image.resize((200, 80))
                name = file.split("/")[-1].split(".")[0]
                image.save(os.path.join("train", f"{name}.png"))

