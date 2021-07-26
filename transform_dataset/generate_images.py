import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import argparse
import torch
from models import Generator


class CustomDataset(Dataset):

    def __init__(self, root, extensions, transform, image_type='IR', cover_type='uncover'):
        self.transform = transform

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
            if cover_type in file:
                files.append(file)

        self.files = files

        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        file = Image.open(file_path).convert('RGB')

        file = self.transform(file)

        return file, file_path


parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', type=str, default='C:\\Users\\Leonardo Capozzi\\Desktop\\VIPcup\\train', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
opt = parser.parse_args()
print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

netG_A2B = netG_A2B.to(device)
netG_B2A = netG_B2A.to(device)

netG_A2B.load_state_dict(torch.load('output_cover2\\netG_A2B.pth', map_location=device))
netG_B2A.load_state_dict(torch.load('output_cover2\\netG_B2A.pth', map_location=device))

netG_A2B.eval()
netG_B2A.eval()

# Dataset loader
transform = transforms.Compose([transforms.Resize((opt.size, opt.size), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

rev_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((160, 120), Image.BICUBIC)])

dataset = CustomDataset(opt.dataroot, ['.png'], transform, image_type='IR', cover_type='uncover')

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, (img, path) in enumerate(dataloader):
    # Set model input
    img = img.to(device)

    with torch.no_grad():
        img_out = netG_A2B(img)

    img_out = img_out[0].cpu() * 0.5 + 0.5
    img_out = rev_transform(img_out)

    path = path[0].replace('uncover', 'cover2_gen')
    path_dir = os.path.split(path)[0]

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    img_out.save(path)
    print(f'{i}/{len(dataset)}')


