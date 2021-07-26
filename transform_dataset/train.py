import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from dataset import CustomDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='C:\\Users\\Leonardo Capozzi\\Desktop\\VIPcup\\train', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
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
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

netG_A2B = netG_A2B.to(device)
netG_B2A = netG_B2A.to(device)
netD_A = netD_A.to(device)
netD_B = netD_B.to(device)

netG_A2B.train()
netG_B2A.train()
netD_A.train()
netD_B.train()

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transform = transforms.Compose([transforms.Resize((int(opt.size * 1.1), int(opt.size * 1.1)), Image.BICUBIC),
                                transforms.RandomCrop(opt.size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = CustomDataset(opt.dataroot, ['.png'], transform, unaligned=True, image_type='IR', cover_type='cover1')

dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (real_A, real_B) in enumerate(dataloader):
        # Set model input
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        ###### Generators A2B and B2A ######

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A.detach().clone())
        pred_fake = netD_A(fake_A)
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5

        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B.detach().clone())
        pred_fake = netD_B(fake_B)
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5

        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()
        ###################################

        # Progress report
        print({'epoch': epoch, 'batch': i, 'loss_G': loss_G.item(), 'loss_G_identity': (loss_identity_A + loss_identity_B).item(),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).item(),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(), 'loss_D': (loss_D_A + loss_D_B).item()})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output\\netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output\\netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output\\netD_A.pth')
    torch.save(netD_B.state_dict(), 'output\\netD_B.pth')
###################################

