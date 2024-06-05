import numpy as np
import os
import torch
from torch.autograd import Variable
from captcha_encoding import decode
from captcha_setting import TEST_DATASET_PATH
from dataset import get_test_item
from model import CNN
from setting import ALL_CHAR_SET, ALL_CHAR_SET_LEN


def test():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    test_data = [os.path.join(TEST_DATASET_PATH, image_file) for image_file in os.listdir(TEST_DATASET_PATH)]

    correct = 0
    total = 0
    for path in test_data:
        images, labels = get_test_item(path)
        image = images.unsqueeze(0)
        vector = Variable(image)
        predict_label = cnn(vector)

        c0 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN * 0:ALL_CHAR_SET_LEN * 1].data.numpy())]
        c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN * 1:ALL_CHAR_SET_LEN * 2].data.numpy())]
        c2 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN * 2:ALL_CHAR_SET_LEN * 3].data.numpy())]
        c3 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN * 3:ALL_CHAR_SET_LEN * 4].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = decode(labels, set_len=ALL_CHAR_SET_LEN)
        total += 1

        if predict_label == true_label:
            correct += 1
        if total % 200 == 0:
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    test()
