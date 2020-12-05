import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
os.system('rm tmp')

import torch
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import Resnet
import VGG
from GPU import get_default_device, to_device, DeviceDataLoader
from Train import fit_one_cycle
import argparse

# initialize input
pars = argparse.ArgumentParser()
pars.add_argument('--model', '-m', type=str, default='res18', metavar='M', help='type of model (res18, res34 or vgg16)')
pars.add_argument('--bsize', '-bs', type=int, default=8, metavar='S', help='batch size of the train data (8, 16, 32, 64 or 128)')
pars.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate of model')
pars.add_argument('--epoch', '-e', type=int, default=200, metavar='E', help='epochs of training')
pars.add_argument('--grad_clip', type=float, default=0.1, metavar='G', help='set grad_clip to avoid grad explosion')
pars.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD', help='set weight_decay')
pars.add_argument('--data', '-d', type=str, default='./JZ20200509/', metavar='D', help='location of data folder')
args = pars.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
data_dir = args.data

# pre processing
tfms = tt.Compose([
    tt.Resize((224, 224)),
    tt.CenterCrop(192),
    tt.ToTensor(),
    tt.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

train_ds = ImageFolder(data_dir + '/train', tfms)
valid_ds = ImageFolder(data_dir + '/test', tfms)

batch_size = args.bsize

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size * 4, num_workers=8, pin_memory=True)

device = get_default_device()
print(device)
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(valid_dl, device)

if args.model == 'res18':
    model = Resnet.resnet18()
elif args.model == 'res34':
    model = Resnet.resnet34()
elif args.model == 'vgg16':
    model = VGG.vgg16_bn()
else:
    exit(-1)

print('Using GPU:' + str(np.argmax(memory_gpu)))
# print(model)
to_device(model, device)

# Training steps
epochs = args.epoch
max_lr = args.lr
grad_clip = args.grad_clip
weight_decay = args.weight_decay
opt_func = torch.optim.Adam
history = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, args, grad_clip=grad_clip,
                        weight_decay=weight_decay, opt_func=opt_func)