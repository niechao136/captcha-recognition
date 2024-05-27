from captcha_generate import generate_captcha_image
from captcha_setting import MAX_CAPTCHA, TRAIN_DATASET_PATH, TEST_DATASET_PATH
from setting import ALL_CHAR_SET


if __name__ == '__main__':
    is_train = True
    if is_train:
        generate_captcha_image(path=TRAIN_DATASET_PATH, max=MAX_CAPTCHA, char_set=ALL_CHAR_SET, is_spp=True)
    else:
        generate_captcha_image(path=TEST_DATASET_PATH, max=MAX_CAPTCHA, char_set=ALL_CHAR_SET, is_spp=True)
