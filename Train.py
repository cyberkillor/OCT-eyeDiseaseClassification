import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

best_accu = 0


def enumerate_robnet_large(num):
    from itertools import product
    arch_list = list(product(['01', '10', '11'], repeat=14))
    arch_list = [list(ele) for ele in arch_list]
    import random
    random.shuffle(arch_list)
    return arch_list[:num]


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def training_step(model, batch, args):
    images, labels = batch
    out = model(images)  # Generate predictions

    if args.model == 'googlenet':
        logits, aux_logits2, aux_logits1 = out

        loss0 = F.cross_entropy(logits, labels)
        loss1 = F.cross_entropy(aux_logits1, labels)
        loss2 = F.cross_entropy(aux_logits2, labels)

        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
    else:
        loss = F.cross_entropy(out, labels)  # Calculate loss

    return loss


def validation_step(model, batch):
    images, labels = batch
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}


def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


def epoch_end(epoch, result, model, args):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss'], result['val_acc']))
    global best_accu

    if args.Save and epoch > 80 and result['val_acc'] > best_accu and result['train_loss'] < 0.005:
        best_accu = result['val_acc']
        print('best_accu: {}'.format(best_accu))
        if args.pretrained is False:
            torch.save({'state_dict': model.state_dict()},
                       './results/{}/Best-Model-bsize{}.pth'.format(args.model, args.bsize))
        else:
            torch.save({'state_dict': model.state_dict()},
                       './results/{}/Best-Model-bsize{}-pt.pth'.format(args.model, args.bsize))


# Traning
@torch.no_grad()
def evaluate(model, val_loader):
    # Tell PyTorch validation start, disable all regularization
    model.eval()
    # Take a Batch loss and Accuracy and Average through all the batches  
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, args,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            # print(batch[0].size())
            loss = training_step(model, batch, args)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(epoch, result, model, args)
        history.append(result)
    return history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
