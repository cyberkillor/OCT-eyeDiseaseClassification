import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
os.system('rm tmp')

import torch_geometric
import torch
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from Models import Resnet, VGG, GoogleNet
from GPU import get_default_device, to_device, DeviceDataLoader
from Train import fit_one_cycle
import argparse
import random

# initialize input
pars = argparse.ArgumentParser()
pars.add_argument('--model', '-m', type=str, default='res18', metavar='M',
                  help='type of model (res18, res34, vgg16 or googlenet)')
pars.add_argument('--bsize', '-bs', type=int, default=8, metavar='S',
                  help='batch size of the train data (8, 16, 32, 64 or 128)')
pars.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate of model')
pars.add_argument('--epoch', '-e', type=int, default=200, metavar='E', help='epochs of training')
pars.add_argument('--grad_clip', type=float, default=0.1, metavar='G', help='set grad_clip to avoid grad explosion')
pars.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD', help='set weight_decay')
pars.add_argument('--data', '-d', type=str, default='./JZ20200509/', metavar='D', help='location of data folder')
pars.add_argument('--Save', default='False', action='store_true',
                  help='whether to save results during training')
pars.add_argument('--pretrained', default='False', action='store_true',
                  help='do finetuning on pretrainded model')
args = pars.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
data_dir = args.data

# set seed
seed = 1111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# pre processing
tfms = tt.Compose([
    tt.Resize(256),
    # tt.Resize((224, 224)),
    tt.CenterCrop(224),
    # tt.Grayscale(num_output_channels=1),
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
    model = Resnet.resnet18(pretrained=args.pretrained)
elif args.model == 'res34':
    model = Resnet.resnet34(pretrained=args.pretrained)
elif args.model == 'vgg16':
    model = VGG.vgg16_bn(pretrained=args.pretrained)
elif args.model == 'googlenet':
    model = GoogleNet.GoogLeNet(num_classes=5, init_weights=True)
else:
    print("Model Fault!")
    exit(-1)

print('Using GPU:' + str(np.argmax(memory_gpu)))
print(model)
to_device(model, device)

# Training steps
epochs = args.epoch
max_lr = args.lr
grad_clip = args.grad_clip
weight_decay = args.weight_decay
opt_func = torch.optim.Adam
history = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, args, grad_clip=grad_clip,
                        weight_decay=weight_decay, opt_func=opt_func)
