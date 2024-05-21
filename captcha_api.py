# coding=utf-8
import os
import numpy as np
import torch
from torch.autograd import Variable
from captcha_cnn_model import CaptchaCNN
from captcha_dataset import get_image
from captcha_setting import ALL_CHAR_SET, ALL_CHAR_SET_LEN
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

app = FastAPI()


@app.post('/uploadfile')
async def uploadfile(file: UploadFile = File(...)):
    cnn = CaptchaCNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    content = file.file.read()
    path = f'upload{os.path.sep}{file.filename}'
    if not os.path.exists('upload'):
        os.makedirs('upload')
    try:
        with open(path, 'wb') as buffer:
            buffer.write(content)
    finally:
        buffer.close()
        img = get_image(path)
        os.remove(path)
    image = img.unsqueeze(0)
    vector = Variable(image)
    predict_label = cnn(vector)

    c0 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN * 0:ALL_CHAR_SET_LEN * 1].data.numpy())]
    c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN * 1:ALL_CHAR_SET_LEN * 2].data.numpy())]
    c2 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN * 2:ALL_CHAR_SET_LEN * 3].data.numpy())]
    c3 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN * 3:ALL_CHAR_SET_LEN * 4].data.numpy())]
    predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
    return {'predict': predict_label}


app.mount('/', StaticFiles(directory='./web/www', html=True))
