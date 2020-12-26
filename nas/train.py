import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
print('Using GPU:' + str(np.argmax(memory_gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
os.system('rm tmp')

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import argparse
from utils import *


def get_args():
    pars = argparse.ArgumentParser()
    pars.add_argument('--data', '-d', type=str, default='/JZ20200509',
                      metavar='D', help='location of data')
    pars.add_argument('--model_num', '-m', type=int, default=0,
                      metavar='M', help='choose to optimaize No. {m} model (0-9)')
    pars.add_argument('--seed', '-s', type=int, default=1111, metavar='S',
                      help='set seed')
    pars.add_argument('--lr', '-lr', type=float, default=0.001,
                      metavar='LR', help='set learning rate')
    pars.add_argument('--load_path', '-lp', type=str, default='./results/',
                      metavar='LP', help='path where you load model')
    pars.add_argument('--init', default=False, action='store_true',
			                    help='whether init the model')
    return pars.parse_args()


def get_data(data_dir):
    tfms = tt.Compose([
        tt.Resize(256),
        tt.CenterCrop(224),
        tt.ToTensor(),
        tt.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    train_ds = ImageFolder(data_dir + '/train', tfms)
    valid_ds = ImageFolder(data_dir + '/test', tfms)
    batch_size = 8
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size * 4, num_workers=8, pin_memory=True)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(valid_dl, device)
    return train_dl, val_dl


if __name__ == '__main__':
    args = get_args()
    print(args.init)
    init_seed(args.seed)
    device = get_default_device()

    max_lr = args.lr
    grad_clip = 0.1
    weight_decay = 3e-4
    opt_func = torch.optim.Adam

    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = p + args.data
    print(data_dir)
    train_dl, valid_dl = get_data(data_dir)

    model = torch.load(args.load_path + 'Model_rank{}.pth'.format(args.model_num))
    if args.init is True:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
    
    to_device(model, device)

    history = fit_one_cycle(100, max_lr, model, train_dl, valid_dl, weight_decay, grad_clip, opt_func)
    print(history)
    if args.init is False:
        torch.save(model, args.load_path + 'Model_rank{}_trained'.format(args.model_num))
    else:
        torch.save(model, args.load_path + 'Model_rank{}_trained_init'.format(args.model_num))
