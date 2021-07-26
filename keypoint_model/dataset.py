import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils import Resize, RandomCrop, RandomHorizontalFlip, ToTensor, show_keypoints


class CustomDataset(Dataset):

    def __init__(self, root, extensions, transform, image_type='IR', cover_types=None):
        if cover_types is None:
            cover_types = ['cover1_gen', 'cover2_gen']

        self.transform = transform
        self.image_type = image_type

        valid_files = []

        for path, subdirs, files in os.walk(root):
            for name in files:
                file_path = os.path.join(path, name)

                if os.path.splitext(file_path)[1] in extensions:
                    valid_files.append(file_path)

        # check correct image type
        valid_files_temp = []

        for file in valid_files:
            if image_type in file:
                valid_files_temp.append(file)

        valid_files = valid_files_temp

        files = []

        for file in valid_files:
            for ct in cover_types:
                if ct in file:
                    files.append(file)
                    break

        self.files = files
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # obtain paths and required values
        image_path = self.files[idx]

        folder = image_path[:image_path.find(os.sep + self.image_type)]

        img_name = os.path.split(image_path)[1]
        img_name = os.path.splitext(img_name)[0]

        num_image = int(img_name[img_name.find('_') + 1:])

        labels_path = os.path.join(folder, 'joints_gt_' + self.image_type + '.mat')

        # load images and labels
        image = Image.open(image_path).convert('RGB')

        if not os.path.exists(labels_path):
            image = self.transform(image)
            return image

        # Obtain the labels for the person in the current image
        labels = sio.loadmat(labels_path)

        # Obtain the labels for the image being opened
        keypoints = labels['joints_gt'][:, :, num_image - 1]
        keypoints = np.moveaxis(keypoints, 0, 1)
        keypoints[:, 0:2] = keypoints[:, 0:2] - 1
        keypoints[:, 2] = 1 - keypoints[:, 2]

        # get bounding box coordinates
        x1 = np.min(keypoints[:, 0])
        x2 = np.max(keypoints[:, 0])
        y1 = np.min(keypoints[:, 1])
        y2 = np.max(keypoints[:, 1])
        box = [[x1, y1, x2, y2]]
        box = torch.as_tensor(box, dtype=torch.float32)
        keypoints = torch.as_tensor([keypoints], dtype=torch.float32)

        # Obtain label (background = 0 and foreground = 1)
        label = [1]
        label = torch.as_tensor(label, dtype=torch.int64)

        target = {'keypoints': keypoints, 'boxes': box, 'labels': label}

        image, target = self.transform((image, target))

        return image, target


def test():
    p = 1.1
    tf = transforms.Compose([Resize((int(160*p), int(120*p))),
                             RandomCrop((160, 120)),
                             RandomHorizontalFlip(),
                             ToTensor()])
    dataset = CustomDataset('C:\\Users\\Leonardo Capozzi\\Desktop\\VIPcup\\train', ['.png'], tf, image_type='IR', cover_types=['cover1_gen', 'cover2_gen'])

    while True:
        d = dataset[3]

        print(d[1])
        show_keypoints(d[0], d[1]['keypoints'][0])


if __name__ == '__main__':
    test()

