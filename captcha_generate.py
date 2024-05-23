# coding=utf-8
import os
import random
from captcha.image import ImageCaptcha
from PIL import Image
from captcha_setting import ALL_CHAR_SET, MAX_CAPTCHA, TRAIN_DATASET_PATH, TEST_DATASET_PATH


# 生成验证码文本
def random_captcha(max=MAX_CAPTCHA, char_set=ALL_CHAR_SET):
    captcha_text = []
    for _ in range(max):
        c = random.choice(char_set)
        captcha_text.append(c)
    return ''.join(captcha_text)


# 生成验证码图片
def gen_captcha_text_and_image(is_spp=False, max=MAX_CAPTCHA, char_set=ALL_CHAR_SET):
    w = int(random.random() * 80) + 120 if is_spp else 160
    h = int(random.random() * 30) + 50 if is_spp else 60
    captcha = ImageCaptcha(width=w, height=h)
    captcha_text = random_captcha(max=max, char_set=char_set)
    captcha_image = Image.open(captcha.generate(captcha_text))
    return captcha_text, captcha_image


def generate_captcha_image(num=10000, path=TRAIN_DATASET_PATH, is_spp=False, max=MAX_CAPTCHA, char_set=ALL_CHAR_SET):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(num):
        text, image = gen_captcha_text_and_image(is_spp=is_spp, max=max, char_set=char_set)
        filename = '%.5d' % i + '_' + text + '.jpg'
        image.save(path + os.path.sep + filename)
        if (i + 1) % 100 == 0:
            print('generate %d images' % (i + 1))


if __name__ == '__main__':
    is_train = True
    if is_train:
        generate_captcha_image(path=TRAIN_DATASET_PATH)
    else:
        generate_captcha_image(path=TEST_DATASET_PATH)
