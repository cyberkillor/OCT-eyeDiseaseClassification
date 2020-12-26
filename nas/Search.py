import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
print('Using GPU:' + str(np.argmax(memory_gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
os.system('rm tmp')
import torch
import argparse
from mmcv import Config
from utils import *
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt

# choose BatchNorm or GroupNorm
from basic_model import Network
# from basic_model_gn import Network


def get_args():
    pars = argparse.ArgumentParser()
    pars.add_argument('--config', '-c', type=str, default='./config-search.py',
                      metavar='C', help='location of the config file')
    pars.add_argument('--save_path', '-sp', type=str, default='./results',
                      metavar='SP', help='path where you save model')
    pars.add_argument('--search_num', '-sn', type=int, default=20,
                      metavar='SN', help='num of models to search')
    return pars.parse_args()


def get_config(args):
    return Config.fromfile(args.config)


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
    cfg = get_config(args)
    init_seed(cfg.seed)
    device = get_default_device()

    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = p + cfg.data
    train_dl, valid_dl = get_data(data_dir)

    arch_code_set = enumerate_robnet_large(args.search_num)
    print("Loaded, Length of the arch_code:", len(arch_code_set))

    Scores = []
    for num, arch_code in enumerate(arch_code_set):
        print("Arch No. ", num)

        net = Network(genotype_list=[arch_code], layers=9)
        to_device(net, device)
        s = eval_all(net, train_dl)
        print("Evaluation Score:", s)
        Scores.append([s, [arch_code]])

        Scores.sort()
        if len(Scores) > 10:
            Scores.pop(0)

    print(Scores)

    # select 10 best models
    for i in range(10):
        arch_code = Scores[len(Scores) - i - 1][1]
        model = Network(genotype_list=arch_code, layers=9)
        to_device(model, device)
        torch.save(model, args.save_path + '/Model_rank{}.pth'.format(i))
