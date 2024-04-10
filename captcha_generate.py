from captcha.image import ImageCaptcha
from PIL import Image
import random
import captcha_setting
import os


# 生成验证码文本
def random_captcha():
    captcha_text = []
    for _ in range(captcha_setting.MAX_CAPTCHA):
        c = random.choice(captcha_setting.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)


# 生成验证码图片
def gen_captcha_text_and_image():
    w = int(random.random() * 100) + 100
    h = int(random.random() * 40) + 40
    captcha = ImageCaptcha(width=w, height=h)
    captcha_text = random_captcha()
    captcha_image = Image.open(captcha.generate(captcha_text))
    return captcha_text, captcha_image


if __name__ == '__main__':
    count = 30
    path = captcha_setting.TRAIN_DATASET_PATH
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        text, image = gen_captcha_text_and_image()
        filename = text + '_' + str(i) + '.png'
        image.save(path + os.path.sep + filename)
