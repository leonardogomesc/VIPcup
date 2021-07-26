import os
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms as transforms


class CustomDataset(Dataset):

    def __init__(self, root, extensions, transform, unaligned=True, image_type='IR', cover_type='cover1'):
        self.transform = transform
        self.unaligned = unaligned

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

        # divide into uncover and cover
        uncover_files = []

        for file in valid_files:
            if 'uncover' in file:
                uncover_files.append(file)

        cover_files = []

        for file in valid_files:
            if cover_type in file:
                cover_files.append(file)

        self.uncover_files = uncover_files
        self.cover_files = cover_files

        self.uncover_files.sort()
        self.cover_files.sort()

    def __len__(self):
        return max(len(self.uncover_files), len(self.cover_files))

    def __getitem__(self, idx):
        uncover_file = self.uncover_files[idx % len(self.uncover_files)]

        if self.unaligned:
            cover_file = self.cover_files[random.randint(0, len(self.cover_files) - 1)]
        else:
            cover_file = self.cover_files[idx % len(self.cover_files)]

        uncover_file = Image.open(uncover_file).convert('RGB')
        cover_file = Image.open(cover_file).convert('RGB')

        uncover_file = self.transform(uncover_file)
        cover_file = self.transform(cover_file)

        return uncover_file, cover_file


def test():
    size = 256
    transform = transforms.Compose([transforms.Resize((int(size * 1.1), int(size * 1.1)), Image.BICUBIC),
                                    transforms.RandomCrop(size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = CustomDataset('C:\\Users\\Leonardo Capozzi\\Desktop\\VIPcup\\train', ['.png'], transform, unaligned=True, image_type='IR', cover_type='cover1')

    for i in dataset:
        transforms.ToPILImage()(i[0] * 0.5 + 0.5).show()
        transforms.ToPILImage()(i[1] * 0.5 + 0.5).show()
        input()


if __name__ == '__main__':
    test()

