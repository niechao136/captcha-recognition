import numpy as np
from captcha_setting import ALL_CHAR_SET_LEN, MAX_CAPTCHA


def char2pos(char):
    k = ord(char) - 48
    if k > 9:
        k = ord(char) - 65 + 10
        if k > 35:
            k = ord(char) - 97 + 26 + 10
            if k > 61:
                raise ValueError('error')
    return k


def encode(text, set_len=ALL_CHAR_SET_LEN):
    vector = np.zeros(set_len * MAX_CAPTCHA, dtype=float)
    for i, c in enumerate(text):
        idx = i * set_len + char2pos(c)
        vector[idx] = 1.0
    return vector


def decode(vector, set_len=ALL_CHAR_SET_LEN):
    char_pos = vector.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % set_len
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        else:
            raise ValueError('error')
        text.append(chr(char_code))

    return "".join(text)


if __name__ == '__main__':
    e = encode("4133")
    print(e, decode(e))
