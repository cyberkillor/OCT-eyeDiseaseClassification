import os
import numpy as np
import torch
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import Resnet3D
from GPU import get_default_device, to_device, DeviceDataLoader
from Train import fit_one_cycle
import argparse

# [1, 3, 19, 224, 224]

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def getArgparse():
    pars = argparse.ArgumentParser()
    pars.add_argument('--model', '-m', type=str, default='res3d', metavar='M',
                      help='No Modification')
    pars.add_argument('--bsize', '-bs', type=int, default=19, metavar='S',
                      help='No Modification')
    pars.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate of model')
    pars.add_argument('--epoch', '-e', type=int, default=200, metavar='E', help='epochs of training')
    pars.add_argument('--grad_clip', type=float, default=0.1, metavar='G', help='set grad_clip to avoid grad explosion')
    pars.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD', help='set weight_decay')
    pars.add_argument('--data', '-d', type=str, default='./JZ20200509/', metavar='D', help='location of data folder')
    return pars.parse_args()


def getData():
    data_dir = './JZ20200509/'
    # pre processing
    tfms = tt.Compose([
        tt.Resize((224, 224)),
        # tt.ColorJitter(),
        tt.ToTensor(),
        tt.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(data_dir + '/train', tfms)
    valid_ds = ImageFolder(data_dir + '/test', tfms)
    batch_size = 19
    train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=8, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(len(train_dl), len(valid_dl))
    # for batch in train_dl:
    #     batch[0].unsqueeze_(0)
    #     print(batch[0].size())
    #     print(batch[1].size())
    #     print(batch[1])
    # for batch in valid_dl:
    #     batch[0].unsqueeze_(0)
    #     print(batch[0].size())
    #     print(batch[1].size())
    #     print(batch[1])

    device = get_default_device()
    print(device)
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(valid_dl, device)
    for img, _ in train_dl:
        img = torch.unsqueeze(img, 0)
    for img, _ in val_dl:
        img.unsqueeze_(0)
    for img, _ in train_dl:
        print(img.size())
        break
    return train_dl, val_dl


def getModel():
    return Resnet3D.generate_model(18, conv1_t_size=19)


if __name__ == '__main__':
    args = getArgparse()
    train, valid = getData()

    model = getModel()

    # Training steps
    epochs = args.epoch
    max_lr = args.lr
    grad_clip = args.grad_clip
    weight_decay = args.weight_decay
    opt_func = torch.optim.Adam
    # history = fit_one_cycle(epochs, max_lr, model, train, valid, args, grad_clip=grad_clip,
    #                         weight_decay=weight_decay, opt_func=opt_func)
    # print(model)

