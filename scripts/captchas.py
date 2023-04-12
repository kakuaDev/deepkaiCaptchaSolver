from captcha.image import ImageCaptcha
from PIL import Image
import os
import random
import string

CHARS_SET = string.digits + string.ascii_uppercase + string.ascii_lowercase
TRAIN_DIR = 'train'
FONTS_DIR = 'fonts'


def remove_output(file_dir):
    if os.path.isdir(file_dir):
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.mkdir(file_dir)


def read_dictionary(dataset_size: int = 50):
    words_list: list = []
    for _ in range(dataset_size):
        word_length = random.randint(2, 10)
        # Remove trailing end line char
        word = "".join(random.sample(CHARS_SET, word_length))
        if word not in words_list:
            words_list.append(word)
            yield word
    del words_list


def gen_captcha_image(image_captcha, text):
    captcha = image_captcha.generate(text)
    captcha_image = Image.open(captcha)
    return captcha_image


def gen_captcha_images(words):
    image_captcha = ImageCaptcha(
        width=200,
        height=80,
        fonts=[os.path.join(FONTS_DIR, font) for font in os.listdir(FONTS_DIR)])
    for word in words:
        chars_list = list(word)
        random.shuffle(chars_list)
        shuffled_word = ''.join(list(map(
            lambda char: char.upper() if random.random() <= 0.5 else char.lower(),
            chars_list
        )))
        captcha_image = gen_captcha_image(image_captcha, shuffled_word)
        captcha_image.save(os.path.join(TRAIN_DIR, shuffled_word + '.png'))


def main():
    # remove_output(TRAIN_DIR)
    words = read_dictionary()
    gen_captcha_images(words)


if __name__ == "__main__":
    main()
