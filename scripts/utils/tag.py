from PIL import Image
import os

_PATH = os.listdir(".")
FILES = []
i = 1
for file in _PATH:
    if file.endswith(".PNG"):
        FILES.append((i, file))
        i = i + 1

for i, file in FILES:
    print(f"{i}/{len(FILES)}")
    img = Image.open(file)
    # img = img.resize((800,320))
    img.show()
    tag = input("Image TAG: ")
    img = img.resize((200, 80))
    img.save(file)
    img.close()
    os.rename(file, f"{tag}.png")

