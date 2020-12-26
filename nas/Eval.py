import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
os.system('rm tmp')

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    classification_report, confusion_matrix
import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from GPU import get_default_device, to_device, DeviceDataLoader
import random

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', '-lp', type=str, default='./results/',
                    metavar='LP', help='path where you load model .../.../')
parser.add_argument('--model_num', '-m', type=int, default=0,
                    metavar='M', help='choose to evaluate No.{0-9} model')
parser.add_argument('--init', default=False, action='store_true',
                    help='whether init the model')
args = parser.parse_args()
device = get_default_device()

# set seed
seed = 1111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print(args.init)
if args.init is True:
    model = torch.load(args.load_path + 'Model_rank{}_trained_init'.format(args.model_num))
else:
    model = torch.load(args.load_path + 'Model_rank{}_trained'.format(args.model_num))

to_device(model, device)
print('Using GPU:' + str(np.argmax(memory_gpu)))

# load data
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/JZ20200509'
tfms = tt.Compose([
    tt.Resize(256),
    tt.CenterCrop(224),
    tt.ToTensor(),
    tt.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])
valid_ds = ImageFolder(data_dir + '/test', tfms)
batch_size = 8
valid_dl = DataLoader(valid_ds, batch_size * 4, num_workers=8, pin_memory=True)
val_dl = DeviceDataLoader(valid_dl, device)


def step(model, batch):
    images, labels = batch
    out = model(images)  # Generate predictions
    preds = torch.max(out, dim=1)[1]

    return {'preds': preds, 'labels': labels, 'out': out}


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [step(model, batch) for batch in val_loader]
    Preds = [x['preds'] for x in outputs]
    Labels = [x['labels'] for x in outputs]
    Outs = [x['out'] for x in outputs]

    Preds = torch.cat(Preds, dim=0).cpu()
    Labels = torch.cat(Labels, dim=0).cpu()
    Outs = torch.cat(Outs, dim=0).cpu()
    Scores = F.softmax(Outs, dim=1)

    print(Preds, Labels, Scores)
    print(Preds.size(), len(Labels), Scores.size())
    print('General Evaluation')

    # Precision | Recall | F1 - score | AUC
    acc_all = accuracy_score(Labels, Preds)
    ap_all = precision_score(Labels, Preds, average='macro')
    ar_all = recall_score(Labels, Preds, average='macro')
    f1_all = f1_score(Labels, Preds, average='macro')
    print(acc_all, ap_all, ar_all, f1_all)

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
    t1 = classification_report(y_true, y_pred,
                               target_names=['AMD', 'DME', 'NM', 'PCV', 'PM'])
    t2 = classification_report(y_true, y_pred, output_dict=True,
                               target_names=['AMD', 'DME', 'NM', 'PCV', 'PM'])
    print(t1)
    print(t2)

    # draw confuse matrix
    classes = ['AMD', 'DME', 'NM', 'PCV', 'PM']
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
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

    if args.init is True:
        fig.savefig(args.load_path + "Best-cm-img{}-init.png".format(args.model_num),
                    dpi=320, format='png')
    else:
        fig.savefig(args.load_path + "Best-cm-img{}.png".format(args.model_num),
                    dpi=320, format='png')

    # cal auc
    roc_ovr = roc_auc_score(Labels, Scores, multi_class='ovr')
    print('--roc-ovr:', roc_ovr)
    roc_ovo = roc_auc_score(Labels, Scores, multi_class='ovo')
    print('--roc-ovo:', roc_ovo)


if __name__ == "__main__":
    evaluate(model, val_dl)
