import argparse

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch

from models import Generator
from dataset import CustomDataset

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

netG_A2B.load_state_dict(torch.load('output\\netG_A2B.pth', map_location=device))
netG_B2A.load_state_dict(torch.load('output\\netG_B2A.pth', map_location=device))

netG_A2B.eval()
netG_B2A.eval()

# Dataset loader
transform = transforms.Compose([transforms.Resize((opt.size, opt.size), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

rev_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((160, 120), Image.BICUBIC)])

dataset = CustomDataset(opt.dataroot, ['.png'], transform, unaligned=False, image_type='IR', cover_type='cover1')

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for i, (real_A, real_B) in enumerate(dataloader):
    # Set model input
    real_A = real_A.to(device)
    real_B = real_B.to(device)

    with torch.no_grad():
        fake_B = netG_A2B(real_A)
        fake_A = netG_B2A(real_B)

    real_A = real_A[0].cpu() * 0.5 + 0.5
    fake_B = fake_B[0].cpu() * 0.5 + 0.5
    real_B = real_B[0].cpu() * 0.5 + 0.5
    fake_A = fake_A[0].cpu() * 0.5 + 0.5

    rev_transform(real_A).show()
    rev_transform(fake_B).show()
    rev_transform(real_B).show()
    rev_transform(fake_A).show()

    input()

