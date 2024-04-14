import torch
import torch.nn as nn
from torch.autograd import Variable
from captcha_cnn_model import CaptchaCNN
from captcha_dataset import get_train_data_loader

num_epochs = 20
batch_size = 100
learning_rate = 0.001


def train():
    cnn = CaptchaCNN()
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images)
            labels = Variable(labels.float())
            predict_labels = cnn(images)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i + 1, "loss:", loss.item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "model.pkl")   # current is model.pkl
                print("save model")
                # print("epoch:", epoch, "step:", i, "loss:", loss.item())

    torch.save(cnn.state_dict(), "model.pkl")   # current is model.pkl
    print("save last model")


if __name__ == '__main__':
    train()
