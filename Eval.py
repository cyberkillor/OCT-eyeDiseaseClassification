import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
print('Using GPU:' + str(np.argmax(memory_gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
os.system('rm tmp')

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, classification_report, confusion_matrix
import argparse
import Resnet
import VGG
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as tt
from GPU import get_default_device, to_device, DeviceDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='res18', metavar='M',
                    help='type of model (res18, res34 or vgg16)')
parser.add_argument('--bsize', '-bs', type=int, default=8, metavar='S',
                    help='batch size of the train data (8, 16, 32, 64 or 128)')
args = parser.parse_args()

device = get_default_device()

if args.model == 'res18':
    model = Resnet.resnet18()
elif args.model == 'res34':
    model = Resnet.resnet34()
elif args.model == 'vgg16':
    model = VGG.vgg16_bn()
else:
    exit(-1)
# print(model)
to_device(model, device)

# load data
data_dir = './JZ20200509/'
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
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(valid_dl, device)

model.load_state_dict(torch.load('./results/{}/Best-Model-bsize{}.pth'.format(args.model, args.bsize)))


def step(model, batch):
    images, labels = batch
    out = model(images)  # Generate predictions
    preds = torch.max(out, dim=1)[1]

    return {'preds': preds, 'labels': labels}


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [step(model, batch) for batch in val_loader]
    Preds = [x['preds'] for x in outputs]
    Labels = [x['labels'] for x in outputs]
    # print(Preds, Labels)

    Preds = torch.cat(Preds, dim=0).cpu()
    Labels = torch.cat(Labels, dim=0).cpu()

    print('General Evaluation')

    y_pred = []
    y_true = []
    for i in range(1900):
        if Preds[i] == 0:
            y_pred.append('AMD')
        elif Preds[i] == 1:
            y_pred.append('DME')
        elif Preds[i] == 2:
            y_pred.append('NM')
        elif Preds[i] == 3:
            y_pred.append('PCV')
        elif Preds[i] == 4:
            y_pred.append('PM')

        if Labels[i] == 0:
            y_true.append('AMD')
        elif Labels[i] == 1:
            y_true.append('DME')
        elif Labels[i] == 2:
            y_true.append('NM')
        elif Labels[i] == 3:
            y_true.append('PCV')
        elif Labels[i] == 4:
            y_true.append('PM')
    t = classification_report(y_true, y_pred, target_names=['AMD', 'DME', 'NM', 'PCV', 'PM'])
    print(t)

    # Precision | Recall | F1 - score
    acc_all = accuracy_score(Labels, Preds)
    ap_all = precision_score(Labels, Preds, average='macro')
    ar_all = recall_score(Labels, Preds, average='macro')
    f1_all = f1_score(Labels, Preds, average='macro')
    print(acc_all, ap_all, ar_all, f1_all)
    # print("accu: {:.4f}, ap: {:.4f}, ar: {:.4f}, f1_score: {:4f}".format(
    #     acc_all, ap_all, ar_all, f1_all))
    
    # draw confuse matrix
    classes = ['AMD', 'DME', 'NM', 'PCV', 'PM']
    cm = confusion_matrix(y_true, y_pred)
    fig,ax = plt.subplots()
    plt.imshow(cm, cmap=plt.cm.Greens)
    indices = range(len(cm))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Pred')
    plt.ylabel('True')
    
    for first_index in range(len(cm)):
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[first_index][second_index])
    fig.savefig("./img/{}/Best-cm-img{}.png".format(args.model, args.bsize),
                dpi=320, format='png')

evaluate(model, val_dl)
