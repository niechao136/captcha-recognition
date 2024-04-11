import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from captcha_encoding import encode
from captcha_setting import TRAIN_DATASET_PATH, TEST_DATASET_PATH


class CaptchaDateset(Dataset):

    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = encode(image_name.split('_')[1].split('.')[0])
        return image, label


trans = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_train_data_loader():
    dataset = CaptchaDateset(TRAIN_DATASET_PATH, transform=trans)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def get_test_data_loader():
    dataset = CaptchaDateset(TEST_DATASET_PATH, transform=trans)
    return DataLoader(dataset, batch_size=1, shuffle=True)
