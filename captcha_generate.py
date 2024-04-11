import os
import random
from captcha.image import ImageCaptcha
from PIL import Image
from captcha_setting import ALL_CHAR_SET, MAX_CAPTCHA, TRAIN_DATASET_PATH, TEST_DATASET_PATH


# 生成验证码文本
def random_captcha():
    captcha_text = []
    for _ in range(MAX_CAPTCHA):
        c = random.choice(ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)


# 生成验证码图片
def gen_captcha_text_and_image():
    # w = int(random.random() * 100) + 100
    # h = int(random.random() * 40) + 4
    w = 160
    h = 60
    captcha = ImageCaptcha(width=w, height=h)
    captcha_text = random_captcha()
    captcha_image = Image.open(captcha.generate(captcha_text))
    return captcha_text, captcha_image


if __name__ == '__main__':
    count = 100000
    path = TRAIN_DATASET_PATH
    # path = TEST_DATASET_PATH
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        text, image = gen_captcha_text_and_image()
        filename = '%.5d' % i + '_' + text + '.jpg'
        image.save(path + os.path.sep + filename)
        if (i + 1) % 100 == 0:
            print('saved %d images' % (i + 1))
