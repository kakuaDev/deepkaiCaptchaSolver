import cv2
import os


FILES = os.listdir("train")

for file in FILES:
    img = cv2.imread(os.path.join("train", file))
    if isinstance(img, type(None)):
        print(file)
        os.remove(os.path.join("train", file))
        continue
    h, w, _ = img.shape
    if h != 80 or w != 200:
        img = cv2.resize(img, (200, 80))
        cv2.imwrite(os.path.join("train", file), img)
